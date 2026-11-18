"""Search tool for the research agents.

``SearchProvider`` is the one interface the agents see: ``search(query, k)``
returns ranked passages. Two adapters implement it:

- ``CorpusSearch``: BM25 (Best Matching 25, a classic keyword-ranking
  formula) over passages of the frozen corpus. Offline, free and
  deterministic, so eval runs are reproducible.
- ``FakeSearch``: canned results for unit tests.

A live web adapter can be added later without touching the agents
(ADR 0015). BM25 rather than embeddings: agents write their own search
queries and can rephrase them, so keyword search is strong enough, and it
needs no embedding model (no quota, no index to build in a database).
"""

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from deep_research.corpus import Document, slugify

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HEADING_RE = re.compile(r"^(={2,})\s*(.+?)\s*\1\s*$")

# Words too common to say anything about relevance. Kept short on purpose:
# BM25's IDF (inverse document frequency) already down-weights common words.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lower-cased alphanumeric words of ``text``, without stopwords."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class Passage(BaseModel):
    """A short stretch of one document: the unit search returns and reports cite.

    ``id`` is ``{document slug}#{n}``, stable as long as the corpus and
    ``passage_words`` do not change.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    section: str
    url: str
    text: str


class SearchHit(BaseModel):
    """A passage with its relevance score for one query (higher is better)."""

    model_config = ConfigDict(frozen=True)

    passage: Passage
    score: float = Field(ge=0)


class SearchProvider(Protocol):
    """Anything that can find passages for a query."""

    name: str

    async def search(self, query: str, k: int) -> list[SearchHit]:
        """Return up to ``k`` hits, best first."""
        ...


def split_passages(doc: Document, target_words: int) -> list[Passage]:
    """Cut ``doc`` into passages of about ``target_words`` words.

    Passages never cross a section heading, and paragraphs are packed
    whole, so a fact is not split from its sentence. A single paragraph
    longer than ``2 * target_words`` is cut on word boundaries.
    """
    slug = slugify(doc.title)
    passages: list[Passage] = []
    section = "Introduction"
    buffer: list[str] = []

    def flush() -> None:
        """Turn the buffered paragraphs into one passage."""
        if buffer:
            n = len(passages)
            passages.append(
                Passage(
                    id=f"{slug}#{n}",
                    title=doc.title,
                    section=section,
                    url=doc.url,
                    text=" ".join(buffer),
                )
            )
            buffer.clear()

    for raw in doc.text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            section = heading.group(2)
            continue
        for piece in _cut_long(line, 2 * target_words):
            if buffer and len(" ".join(buffer).split()) + len(piece.split()) > target_words:
                flush()
            buffer.append(piece)
    flush()
    return passages


def _cut_long(paragraph: str, limit: int) -> list[str]:
    """``paragraph`` in pieces of at most ``limit`` words."""
    words = paragraph.split()
    return [" ".join(words[i : i + limit]) for i in range(0, len(words), limit)]


class BM25Index:
    """BM25 ranking over a fixed list of passages, held in memory.

    For a query, each passage scores the sum over query terms of
    ``idf(term) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len))``.
    ``k1`` caps how much repeating a term helps; ``b`` stops long passages
    winning just by being long. 1.5 and 0.75 are the textbook defaults.
    The title and section are indexed with the text, so a query naming an
    article ("Chandrayaan-3 landing") finds its passages.
    """

    def __init__(self, passages: Sequence[Passage], *, k1: float = 1.5, b: float = 0.75) -> None:
        """Index ``passages``; this takes about a second for the full corpus."""
        self.passages = list(passages)
        self._k1 = k1
        self._b = b
        self._tfs = [Counter(tokenize(f"{p.title} {p.section} {p.text}")) for p in self.passages]
        self._lengths = [sum(tf.values()) for tf in self._tfs]
        self._avg_len = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        df: Counter[str] = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        n = len(self.passages)
        # The "+1" inside the log keeps IDF positive even for a term in
        # more than half the passages (the Lucene variant of BM25).
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def top(self, query: str, k: int) -> list[SearchHit]:
        """The ``k`` best passages for ``query``; ties keep corpus order."""
        terms = [t for t in dict.fromkeys(tokenize(query)) if t in self._idf]
        if not terms or k <= 0:
            return []
        scores: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tfs):
            score = 0.0
            norm = self._k1 * (1 - self._b + self._b * self._lengths[i] / self._avg_len)
            for term in terms:
                freq = tf.get(term, 0)
                if freq:
                    score += self._idf[term] * freq * (self._k1 + 1) / (freq + norm)
            if score > 0:
                scores.append((score, i))
        scores.sort(key=lambda s: (-s[0], s[1]))
        return [SearchHit(passage=self.passages[i], score=s) for s, i in scores[:k]]


class CorpusSearch:
    """``SearchProvider`` over the frozen corpus, using ``BM25Index``."""

    name = "corpus-bm25"

    def __init__(self, documents: Sequence[Document], passage_words: int) -> None:
        """Split every document into passages and index them."""
        passages = [p for doc in documents for p in split_passages(doc, passage_words)]
        self.index = BM25Index(passages)

    async def search(self, query: str, k: int) -> list[SearchHit]:
        """Return up to ``k`` hits, best first. CPU-only and fast, so no thread hop."""
        return self.index.top(query, k)


class FakeSearch:
    """``SearchProvider`` for tests: returns the same hits for every query.

    ``queries`` records every query it was asked, so tests can check what
    an agent searched for.
    """

    name = "fake"

    def __init__(self, hits: Sequence[SearchHit]) -> None:
        """Serve ``hits`` (truncated to ``k``) for every query."""
        self.hits = list(hits)
        self.queries: list[str] = []

    async def search(self, query: str, k: int) -> list[SearchHit]:
        """Record ``query`` and return the first ``k`` canned hits."""
        self.queries.append(query)
        return self.hits[:k]
