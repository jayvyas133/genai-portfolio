"""Split documents into section-aware, overlapping chunks.

Chunks never cross a section heading, so each covers one topic and can
say where it came from (``CUDA > History``). Long sections are cut into
word windows that overlap, so a sentence cut at one window's edge still
appears whole in the neighbouring chunk.

Words stand in for tokens: BGE reads at most 512 tokens (about 370
technical English words). 200-word chunks fit without a tokenizer (ADR 0009).
"""

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from documind.corpus import Document

INTRO_SECTION = "Introduction"
# Boilerplate with no answerable content; their subsections are dropped too.
SKIPPED_SECTIONS = frozenset(
    {
        "References",
        "See also",
        "External links",
        "Further reading",
        "Notes",
        "Sources",
        "Bibliography",
        "Citations",
        "Footnotes",
    }
)
# "== Title ==" is level 0, "=== Title ===" level 1, and so on.
_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1$")


@dataclass(frozen=True)
class Section:
    """Body text under one heading path, e.g. ``History > Early years``."""

    path: str
    text: str


def with_header(title: str, section: str, text: str) -> str:
    """Prefix ``text`` with a ``Title > Section`` line.

    The one place this format is defined: chunks are embedded with it and
    reranked with it, so both models see the same context. Changing it
    changes every content hash, so every chunk would be re-embedded.
    """
    return f"{title} > {section}\n{text}"


class Chunk(BaseModel):
    """One retrievable piece of a document.

    ``ordinal`` is the chunk's position within its document.
    ``embed_text`` is what gets embedded; ``content_hash`` identifies it,
    so unchanged chunks are never embedded twice.
    """

    model_config = ConfigDict(frozen=True)

    document_title: str
    ordinal: int = Field(ge=0)
    section: str
    text: str = Field(min_length=1)

    @property
    def embed_text(self) -> str:
        """The text with a ``Title > Section`` header for context."""
        return with_header(self.document_title, self.section, self.text)

    @property
    def content_hash(self) -> str:
        """SHA-256 (Secure Hash Algorithm) hex digest of ``embed_text``."""
        return hashlib.sha256(self.embed_text.encode("utf-8")).hexdigest()


def _raw_sections(text: str) -> Iterator[tuple[list[str], str]]:
    """Yield ``(heading path, body)`` pairs in order; the lead has path ``[]``."""
    path: list[str] = []
    body: list[str] = []
    for line in text.splitlines():
        match = _HEADING.match(line.strip())
        if match is None:
            body.append(line)
            continue
        yield path, "\n".join(body)
        level = len(match.group(1)) - 2
        path = [*path[:level], match.group(2)]
        body = []
    yield path, "\n".join(body)


def split_sections(text: str) -> list[Section]:
    """Split plain text into sections with heading paths.

    Sections with no body (a heading directly followed by a
    subheading) and boilerplate sections are left out.
    """
    return [
        Section(" > ".join(path) or INTRO_SECTION, body.strip())
        for path, body in _raw_sections(text)
        if body.strip() and not (path and path[0] in SKIPPED_SECTIONS)
    ]


def window_words(text: str, size: int, overlap: int) -> list[str]:
    """Cut ``text`` into windows of ``size`` words sharing ``overlap`` words.

    Text of at most ``size`` words becomes a single window.
    Whitespace is normalised to single spaces.

    Raises:
        ValueError: If ``size`` < 1 or ``overlap`` is not in ``[0, size)``.
    """
    if size < 1 or not 0 <= overlap < size:
        msg = f"need size >= 1 and 0 <= overlap < size, got {size=}, {overlap=}"
        raise ValueError(msg)
    words = text.split()
    step = size - overlap
    windows: list[str] = []
    for start in range(0, len(words), step):
        windows.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return windows


def chunk_document(doc: Document, size: int, overlap: int) -> list[Chunk]:
    """Split ``doc`` into chunks numbered 0, 1, 2, ... in reading order."""
    pieces = [
        (section.path, window)
        for section in split_sections(doc.text)
        for window in window_words(section.text, size, overlap)
    ]
    return [
        Chunk(document_title=doc.title, ordinal=i, section=path, text=text)
        for i, (path, text) in enumerate(pieces)
    ]
