"""The research report: the one output type every research system returns.

The baseline (M1) and the agent graph (M2 onwards) both produce a
``Report``, so the same eval scores them and their numbers are directly
comparable.

Citations are ``[n]`` markers in the text, where ``n`` numbers
``sources`` from 1. The markers are what a reader sees, so there is no
separate citation list that could disagree with the text.
"""

import re
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field

from deep_research.search import Passage

_MARKER_RE = re.compile(r"\[(\d+)\]")
# A sentence ends at . ! or ? followed by optional markers, then whitespace.
# Markers are kept with the sentence before them ("... in 2019 [2]. Next").
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])((?:\s*\[\d+\])*)\s+")


class Report(BaseModel):
    """A written answer to a research question, with the sources it may cite."""

    model_config = ConfigDict(frozen=True)

    question: str
    text: str
    sources: list[Passage] = Field(default_factory=list)
    # The sub-questions the report was planned around (empty for the baseline).
    outline: list[str] = Field(default_factory=list)


Researcher = Callable[[str], Awaitable[Report]]
"""Anything that turns a question into a report: the thing the eval runs."""


class Claim(BaseModel):
    """One sentence of a report and the source numbers it cites."""

    model_config = ConfigDict(frozen=True)

    sentence: str
    citations: list[int]


def markers(text: str) -> list[int]:
    """Every ``[n]`` number in ``text``, in order, repeats included."""
    return [int(n) for n in _MARKER_RE.findall(text)]


def split_claims(text: str) -> list[Claim]:
    """Split a report into sentences, each with the markers it carries.

    Markdown headings and bullet markers are treated as text; a heading
    line without a full stop simply joins the sentence after it, which
    is harmless for scoring.
    """
    claims: list[Claim] = []
    for paragraph in re.split(r"\n\s*\n|\n(?=\s*[-*#])", text):
        flat = " ".join(paragraph.split())
        if not flat:
            continue
        # Put the trailing markers back on the sentence they follow.
        parts = _SENTENCE_END_RE.split(flat)
        sentences: list[str] = []
        for i in range(0, len(parts), 2):
            trailing = parts[i + 1] if i + 1 < len(parts) else ""
            sentence = (parts[i] + trailing).strip()
            if sentence:
                sentences.append(sentence)
        for sentence in sentences:
            cited = list(dict.fromkeys(markers(sentence)))
            claims.append(Claim(sentence=sentence, citations=cited))
    return claims


def invalid_markers(report: Report) -> list[int]:
    """Marker numbers that point at no source (outside ``1..len(sources)``)."""
    return [n for n in markers(report.text) if not 1 <= n <= len(report.sources)]


def strip_markers(text: str) -> str:
    """``text`` without ``[n]`` markers, for showing a sentence to the judge."""
    return " ".join(re.sub(r"\s*\[\d+\]", "", text).split())
