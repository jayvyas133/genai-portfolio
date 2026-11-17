"""A scripted, offline provider for tests and local development.

``FakeProvider`` needs no network or API key. It replays a scripted sequence
of replies and errors, so tests can exercise the client's retry logic
deterministically. It also embeds and reranks offline:

* ``embed`` uses the hashing trick: each word is hashed into one of
  ``dimensions`` buckets with a +/- sign, then the vector is L2-normalized.
  Texts that share words get a high cosine similarity, so retrieval tests
  produce meaningful rankings without a real model.
* ``rerank`` scores a document by the fraction of distinct query words it
  contains.
"""

import hashlib
import math
import re
from collections.abc import Sequence

from aicore.models import (
    EmbedRequest,
    EmbedResponse,
    LLMRequest,
    LLMResponse,
    RerankRequest,
    RerankResponse,
    RerankResult,
    Usage,
)

Outcome = str | Exception
"""One scripted result: a ``str`` is returned as reply text, an ``Exception`` is raised."""

_WORD_RE = re.compile(r"\w+")


def _words(text: str) -> list[str]:
    """Lower-case word tokens of ``text`` (punctuation is dropped)."""
    return _WORD_RE.findall(text.lower())


def _hashed_vector(text: str, dimensions: int) -> list[float]:
    """Embed ``text`` with the hashing trick and L2-normalize the result.

    ``blake2b`` is used instead of the built-in ``hash()`` because Python
    salts string hashes per process, which would make vectors change between
    runs. A text with no words maps to the zero vector.
    """
    vector = [0.0] * dimensions
    for word in _words(text):
        value = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "big")
        sign = -1.0 if value >> 63 else 1.0  # top bit picks the sign
        vector[value % dimensions] += sign  # low bits pick the bucket
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else vector


class FakeProvider:
    """In-memory provider that replays scripted outcomes and embeds/reranks offline.

    Each call to ``complete`` consumes the next outcome from the script. Once
    the script is exhausted, every call returns ``default_reply``. ``embed``
    and ``rerank`` first raise any errors left in their own error scripts,
    then compute deterministic results.

    Attributes:
        calls: Every completion request received, in order.
        embed_calls: Every embedding request received, in order.
        rerank_calls: Every rerank request received, in order.
    """

    name = "fake"

    def __init__(
        self,
        script: Sequence[Outcome] = (),
        default_reply: str = "fake reply",
        model: str = "fake-model",
        *,
        embed_errors: Sequence[Exception] = (),
        rerank_errors: Sequence[Exception] = (),
        dimensions: int = 64,
    ) -> None:
        """Create a provider that will replay ``script`` in order.

        Args:
            script: Outcomes to return or raise, one per ``complete`` call.
            default_reply: Reply text used after the script runs out.
            model: Model name reported when the request does not set one.
            embed_errors: Exceptions raised by the first ``embed`` calls, in order.
            rerank_errors: Exceptions raised by the first ``rerank`` calls, in order.
            dimensions: Length of every embedding vector.
        """
        self._script = list(script)  # copy, so the caller's list is never mutated
        self._default_reply = default_reply
        self._model = model
        self._embed_errors = list(embed_errors)
        self._rerank_errors = list(rerank_errors)
        self._dimensions = dimensions
        self.calls: list[LLMRequest] = []
        self.embed_calls: list[EmbedRequest] = []
        self.rerank_calls: list[RerankRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Record the request, then return or raise the next scripted outcome.

        Token counts are approximated by word counts. That is good enough for
        tests and keeps this class free of tokenizer dependencies.

        Raises:
            Exception: Whatever exception is next in the script.
        """
        self.calls.append(request)
        outcome = self._script.pop(0) if self._script else self._default_reply
        if isinstance(outcome, Exception):
            raise outcome
        input_tokens = sum(len(m.content.split()) for m in request.messages)
        return LLMResponse(
            text=outcome,
            model=request.model or self._model,
            usage=Usage(input_tokens=input_tokens, output_tokens=len(outcome.split())),
        )

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        """Record the request, raise the next scripted error if any, else embed.

        ``input_type`` is ignored: this fake model is symmetric.

        Raises:
            Exception: Whatever exception is next in ``embed_errors``.
        """
        self.embed_calls.append(request)
        if self._embed_errors:
            raise self._embed_errors.pop(0)
        return EmbedResponse(
            vectors=[_hashed_vector(text, self._dimensions) for text in request.texts],
            model=request.model or self._model,
            usage=Usage(
                input_tokens=sum(len(_words(t)) for t in request.texts),
                output_tokens=0,
            ),
        )

    async def rerank(self, request: RerankRequest) -> RerankResponse:
        """Record the request, raise the next scripted error if any, else rerank.

        Score is the fraction of distinct query words found in the document.
        Ties keep the original document order (``sorted`` is stable).

        Raises:
            Exception: Whatever exception is next in ``rerank_errors``.
        """
        self.rerank_calls.append(request)
        if self._rerank_errors:
            raise self._rerank_errors.pop(0)
        query_words = set(_words(request.query))
        results = [
            RerankResult(
                index=i,
                score=len(query_words & set(_words(doc))) / len(query_words)
                if query_words
                else 0.0,
            )
            for i, doc in enumerate(request.documents)
        ]
        results = sorted(results, key=lambda r: r.score, reverse=True)[: request.top_n]
        return RerankResponse(
            results=results,
            model=request.model or self._model,
            usage=Usage(
                input_tokens=len(_words(request.query))
                + sum(len(_words(d)) for d in request.documents),
                output_tokens=0,
            ),
        )
