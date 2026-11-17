"""Tests for the golden dataset models and ``load_golden``."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from aicore.evals.dataset import DatasetError
from documind.chunking import chunk_document
from documind.config import DocumindSettings
from documind.corpus import load_documents
from documind.golden import Evidence, GoldenCase, load_golden, normalize, validate_quotes
from pydantic import ValidationError

EVIDENCE = {"title": "CUDA", "quote": "released in 2007"}
GOLDEN = Path(__file__).resolve().parents[1] / "evals" / "golden.jsonl"


def case(**overrides: Any) -> dict[str, Any]:
    """A valid factual dev case as a dict, with ``overrides`` applied."""
    base: dict[str, Any] = {
        "id": "c1",
        "question": "When was CUDA released?",
        "reference_answer": "2007",
        "evidence": [EVIDENCE],
        "kind": "factual",
        "split": "dev",
    }
    return base | overrides


def write_jsonl(path: Path, *rows: dict[str, Any] | str) -> Path:
    """Write ``rows`` (dicts as JSON, strings verbatim) one per line."""
    lines = [r if isinstance(r, str) else json.dumps(r) for r in rows]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_normalize_collapses_whitespace_and_case() -> None:
    """Line breaks and capitals in a copied quote do not stop a match."""
    assert normalize("  Released\n in   2007 ") == "released in 2007"


def test_valid_case_parses() -> None:
    """A well-formed factual case round-trips into the model."""
    parsed = GoldenCase.model_validate(case())

    assert parsed.evidence == [Evidence(**EVIDENCE)]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"evidence": []}, "factual cases need at least one"),
        ({"kind": "multi-hop"}, "multi-hop cases need at least two"),
        ({"kind": "unanswerable"}, "unanswerable cases must have no evidence"),
        ({"split": "train"}, "split"),
        ({"extra": 1}, "extra"),
    ],
)
def test_invalid_cases_are_rejected(overrides: dict[str, Any], message: str) -> None:
    """Evidence must match ``kind``; unknown splits and fields fail loudly."""
    with pytest.raises(ValidationError, match=message):
        GoldenCase.model_validate(case(**overrides))


def test_unanswerable_without_evidence_is_valid() -> None:
    """An unanswerable case carries no evidence."""
    parsed = GoldenCase.model_validate(case(kind="unanswerable", evidence=[]))

    assert parsed.evidence == []


def test_long_quotes_are_rejected() -> None:
    """A quote longer than one chunk's overlap could never match a chunk."""
    with pytest.raises(ValidationError, match="31 words"):
        Evidence(title="CUDA", quote=" ".join(["word"] * 31))


def test_load_filters_by_split(tmp_path: Path) -> None:
    """``split`` keeps only matching cases, in file order."""
    path = write_jsonl(
        tmp_path / "g.jsonl", case(id="a"), "", case(id="b", split="test"), case(id="c")
    )

    assert [c.id for c in load_golden(path)] == ["a", "b", "c"]
    assert [c.id for c in load_golden(path, "dev")] == ["a", "c"]
    assert [c.id for c in load_golden(path, "test")] == ["b"]


def test_load_ignores_bom(tmp_path: Path) -> None:
    """A BOM written by a Windows editor does not break the first line."""
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps(case()), encoding="utf-8-sig")

    assert len(load_golden(path)) == 1


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ((case(), "{not json"), r"g\.jsonl:2: "),
        ((case(), case()), "duplicate id 'c1' \\(first used on line 1\\)"),
        ((case(evidence=[]),), r"g\.jsonl:1: line: Value error, factual"),
        (("",), "no cases"),
    ],
)
def test_load_errors_name_the_line(
    tmp_path: Path, rows: tuple[dict[str, Any] | str, ...], message: str
) -> None:
    """Every problem is a ``DatasetError`` that points at the bad line."""
    path = write_jsonl(tmp_path / "g.jsonl", *rows)

    with pytest.raises(DatasetError, match=message):
        load_golden(path)


def test_load_empty_split_is_an_error(tmp_path: Path) -> None:
    """Asking for a split with no cases fails instead of scoring nothing."""
    path = write_jsonl(tmp_path / "g.jsonl", case())

    with pytest.raises(DatasetError, match="no cases in split 'test'"):
        load_golden(path, "test")


def test_missing_file_is_a_dataset_error(tmp_path: Path) -> None:
    """A wrong path is reported as a ``DatasetError``, not ``OSError``."""
    with pytest.raises(DatasetError, match="cannot read dataset"):
        load_golden(tmp_path / "missing.jsonl")


def test_validate_quotes_accepts_quotes_inside_a_chunk() -> None:
    """Case and whitespace differences between quote and chunk are ignored."""
    cases = [
        GoldenCase.model_validate(case(evidence=[{"title": "CUDA", "quote": "Released\nin 2007"}]))
    ]

    assert validate_quotes(cases, {"CUDA": ["CUDA was released in 2007 by Nvidia."]}) == []


def test_validate_quotes_reports_missing_quotes_and_unknown_titles() -> None:
    """A quote no chunk holds, or a title not in the corpus, is one problem each."""
    cases = [
        GoldenCase.model_validate(case(id="a", evidence=[{"title": "CUDA", "quote": "in 2009"}])),
        GoldenCase.model_validate(case(id="b", evidence=[{"title": "Nope", "quote": "x"}])),
    ]

    problems = validate_quotes(cases, {"CUDA": ["released in 2007"]})

    assert problems == [
        "a: quote not found in any 'CUDA' chunk: 'in 2009'",
        "b: unknown document 'Nope'",
    ]


def test_validate_quotes_rejects_a_quote_split_across_chunks() -> None:
    """A quote spanning two chunks can never be retrieved whole."""
    cases = [
        GoldenCase.model_validate(case(evidence=[{"title": "CUDA", "quote": "2007 by Nvidia"}]))
    ]

    assert len(validate_quotes(cases, {"CUDA": ["released in 2007", "by Nvidia"]})) == 1


def test_committed_golden_set_is_valid_and_balanced() -> None:
    """The real dataset loads, keeps its planned mix, and every quote is retrievable.

    Uses the default chunk settings, so changing chunking in a way that
    breaks a quote fails here rather than silently lowering eval scores.
    """
    settings = DocumindSettings(_env_file=None)
    cases = load_golden(GOLDEN)
    chunks = {
        doc.title: [
            c.text for c in chunk_document(doc, settings.chunk_words, settings.chunk_overlap_words)
        ]
        for doc in load_documents(settings.data_dir / "raw")
    }

    mix = Counter((c.split, c.kind) for c in cases)
    assert sum(n for (split, _), n in mix.items() if split == "dev") == 40
    assert sum(n for (split, _), n in mix.items() if split == "test") == 20
    assert mix[("dev", "unanswerable")] + mix[("test", "unanswerable")] == 6
    assert validate_quotes(cases, chunks) == []


def test_validate_quotes_checks_alternatives_too() -> None:
    """A wrong alternative quote is reported like a wrong main quote."""
    evidence = {
        "title": "CUDA",
        "quote": "released in 2007",
        "alternatives": [{"title": "Nvidia", "quote": "CUDA launched in 2006"}],
    }
    cases = [GoldenCase.model_validate(case(evidence=[evidence]))]
    chunks = {"CUDA": ["released in 2007"], "Nvidia": ["CUDA launched in 2007"]}

    assert validate_quotes(cases, chunks) == [
        "c1: quote not found in any 'Nvidia' chunk: 'CUDA launched in 2006'"
    ]
