"""The golden dataset: questions with known answers and known evidence.

One case per line of a JSONL (JSON Lines) file, for example::

    {"id": "cuda-001", "question": "When was CUDA first released?",
     "reference_answer": "In 2007.", "kind": "factual", "split": "dev",
     "evidence": [{"title": "CUDA", "quote": "initially released in 2007"}]}

Evidence is a document title plus a short verbatim quote, never a chunk
id. Chunk ids change whenever chunking changes; a quote does not, so the
dataset survives re-chunking and retrieval runs stay comparable.
"""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal, Self

from aicore.evals.dataset import DatasetError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Kind = Literal["factual", "multi-hop", "unanswerable"]
Split = Literal["dev", "test"]

# A quote must fit inside one chunk. With 40 words of overlap, any run of
# up to 41 words appears whole in some chunk; 30 leaves a safety margin.
MAX_QUOTE_WORDS = 30


def normalize(text: str) -> str:
    """Lower-case ``text`` and collapse whitespace, for quote matching.

    The chunker joins words with single spaces, so a quote copied from
    the raw document (with line breaks) still matches its chunk.
    """
    return " ".join(text.split()).lower()


class Source(BaseModel):
    """A place a fact is stated: a document title and a short verbatim quote."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str = Field(min_length=1)
    quote: str = Field(min_length=1)

    @model_validator(mode="after")
    def _quote_fits_in_a_chunk(self) -> Self:
        """Reject quotes too long to appear whole inside one chunk."""
        words = len(self.quote.split())
        if words > MAX_QUOTE_WORDS:
            msg = f"quote has {words} words; keep it to {MAX_QUOTE_WORDS} or fewer"
            raise ValueError(msg)
        return self


class Evidence(Source):
    """A fact the answer needs, and every place the corpus states it.

    ``title`` and ``quote`` are the main location. ``alternatives`` lists
    other documents that state the same fact; retrieving any of them
    counts. They are added when reviewing misses shows retrieval found
    the fact somewhere the label did not list (an incomplete label, not a
    retrieval failure).
    """

    alternatives: list[Source] = Field(default_factory=list)

    @property
    def sources(self) -> list[Source]:
        """The main location followed by the alternatives."""
        return [Source(title=self.title, quote=self.quote), *self.alternatives]


class GoldenCase(BaseModel):
    """One question with its reference answer and supporting evidence.

    ``kind`` decides what evidence is required:

    - ``factual``: at least one evidence passage.
    - ``multi-hop``: at least two passages, since the answer combines them.
    - ``unanswerable``: no evidence; the corpus does not contain the
      answer and the system should refuse (scored in M5).

    ``split`` keeps tuning honest: tune on ``dev``, report ``test`` once.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    evidence: list[Evidence] = Field(default_factory=list)
    kind: Kind
    split: Split

    @model_validator(mode="after")
    def _evidence_matches_kind(self) -> Self:
        """Check the number of evidence passages against ``kind``."""
        count = len(self.evidence)
        if self.kind == "unanswerable" and count:
            msg = "unanswerable cases must have no evidence"
            raise ValueError(msg)
        if self.kind == "factual" and count < 1:
            msg = "factual cases need at least one evidence passage"
            raise ValueError(msg)
        if self.kind == "multi-hop" and count < 2:
            msg = "multi-hop cases need at least two evidence passages"
            raise ValueError(msg)
        return self


def load_golden(path: Path, split: Split | None = None) -> list[GoldenCase]:
    """Read the golden cases in ``path``, optionally only one ``split``.

    Blank lines are skipped and a leading BOM (Byte Order Mark) is
    ignored. The whole file is validated before filtering, so a broken
    test case fails a dev run too.

    Raises:
        DatasetError: If the file cannot be read, a line is invalid
            (message starts with ``path:line``), two cases share an
            ``id``, or no cases remain after filtering.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as err:
        raise DatasetError(f"{path}: cannot read dataset: {err.strerror}") from err

    cases: list[GoldenCase] = []
    first_line_of: dict[str, int] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        case = _parse_line(path, lineno, line)
        if case.id in first_line_of:
            raise DatasetError(
                f"{path}:{lineno}: duplicate id {case.id!r} "
                f"(first used on line {first_line_of[case.id]})"
            )
        first_line_of[case.id] = lineno
        cases.append(case)

    selected = [c for c in cases if split is None or c.split == split]
    if not selected:
        # An empty golden set would make every metric meaningless.
        raise DatasetError(f"{path}: no cases" + (f" in split {split!r}" if split else ""))
    return selected


def validate_quotes(
    cases: Iterable[GoldenCase], chunks_by_title: Mapping[str, Sequence[str]]
) -> list[str]:
    """Return one problem message per quote (main or alternative) no chunk contains.

    ``chunks_by_title`` maps each document title to its chunk texts.
    Checking chunks, not whole documents, catches quotes that span a
    section heading or sit in a dropped section (such as References):
    retrieval could never return them, so the case could never pass.

    An empty list means every quote is retrievable.
    """
    normalized = {
        title: [normalize(text) for text in texts] for title, texts in chunks_by_title.items()
    }
    problems: list[str] = []
    for case in cases:
        for source in (s for evidence in case.evidence for s in evidence.sources):
            texts = normalized.get(source.title)
            if texts is None:
                problems.append(f"{case.id}: unknown document {source.title!r}")
            elif not any(normalize(source.quote) in text for text in texts):
                problems.append(
                    f"{case.id}: quote not found in any {source.title!r} chunk: {source.quote!r}"
                )
    return problems


def _parse_line(path: Path, lineno: int, line: str) -> GoldenCase:
    """Validate one line as a ``GoldenCase``, reporting only the first error."""
    try:
        return GoldenCase.model_validate_json(line)
    except ValidationError as err:
        first = err.errors(include_input=False)[0]
        where = ".".join(str(part) for part in first["loc"]) or "line"
        raise DatasetError(
            f"{path}:{lineno}: {where}: {first['msg']} ({err.error_count()} error(s))"
        ) from err
