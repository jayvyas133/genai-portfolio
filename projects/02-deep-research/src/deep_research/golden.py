"""The golden dataset: research questions and the key facts a good report states.

One case per line of ``evals/golden.jsonl``. Each key fact carries the
corpus document and a short verbatim quote that states it. The quote
does two jobs: ``check_quotes`` proves the fact really is in the corpus
(so a missed fact is the system's fault, not the dataset's), and the eval
uses it to tell a retrieval miss (no retrieved passage holds the quote)
from a writing miss (the passage was found but the report left it out).
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from aicore.evals.dataset import DatasetError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from deep_research.corpus import Document

Split = Literal["dev", "test"]


def normalize(text: str) -> str:
    """Lower-case ``text`` and collapse whitespace, for quote matching."""
    return " ".join(text.split()).lower()


class KeyFact(BaseModel):
    """One fact the report should state, and where the corpus states it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fact: str = Field(min_length=1)
    title: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class GoldenCase(BaseModel):
    """A research question that needs several facts, usually from several documents."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    split: Split
    question: str = Field(min_length=1)
    key_facts: list[KeyFact] = Field(min_length=2)

    @property
    def expected_titles(self) -> list[str]:
        """Documents the key facts come from, in first-mention order."""
        return list(dict.fromkeys(f.title for f in self.key_facts))


def load_golden(path: Path, split: Split | None = None) -> list[GoldenCase]:
    """Load the cases in ``path``, optionally only one ``split``.

    Raises:
        DatasetError: If a line is not a valid case or an id repeats,
            naming the file and line.
    """
    cases: list[GoldenCase] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = GoldenCase.model_validate_json(line)
        except ValidationError as err:
            raise DatasetError(f"{path}:{lineno}: {err.error_count()} error(s): {err}") from err
        if case.id in seen:
            raise DatasetError(f"{path}:{lineno}: duplicate id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    return [c for c in cases if split is None or c.split == split]


def check_quotes(cases: Sequence[GoldenCase], documents: Sequence[Document]) -> list[str]:
    """Problems with the dataset's quotes: unknown titles or quotes not in their document.

    Returns an empty list when every quote is found verbatim (up to case
    and whitespace) in the document it names.
    """
    texts = {d.title: normalize(d.text) for d in documents}
    problems: list[str] = []
    for case in cases:
        for i, fact in enumerate(case.key_facts, start=1):
            if fact.title not in texts:
                problems.append(f"{case.id} fact {i}: unknown title {fact.title!r}")
            elif normalize(fact.quote) not in texts[fact.title]:
                problems.append(f"{case.id} fact {i}: quote not found in {fact.title!r}")
    return problems
