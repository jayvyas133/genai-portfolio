"""Tests for the golden dataset loader and its corpus check."""

import json
from pathlib import Path

import pytest
from aicore.evals.dataset import DatasetError
from deep_research.config import DeepResearchSettings
from deep_research.corpus import Document, load_documents
from deep_research.golden import GoldenCase, KeyFact, check_quotes, load_golden


def _line(case_id: str, split: str = "dev") -> str:
    facts = [{"fact": "f", "title": "Luna 25", "quote": "crashed"}] * 2
    return json.dumps({"id": case_id, "split": split, "question": "q", "key_facts": facts})


def test_load_golden_filters_by_split(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(f"{_line('a')}\n\n{_line('b', 'test')}\n", encoding="utf-8")
    assert [c.id for c in load_golden(path)] == ["a", "b"]
    assert [c.id for c in load_golden(path, "test")] == ["b"]


def test_load_golden_rejects_duplicate_ids_and_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(f"{_line('a')}\n{_line('a')}\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="duplicate id"):
        load_golden(path)
    path.write_text('{"id": "a"}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match=r"golden.jsonl:1"):
        load_golden(path)


def test_expected_titles_are_unique_in_order() -> None:
    facts = [KeyFact(fact="f", title=t, quote="q") for t in ("B", "A", "B")]
    assert GoldenCase(id="x", split="dev", question="q", key_facts=facts).expected_titles == [
        "B",
        "A",
    ]


def test_check_quotes_reports_unknown_titles_and_missing_quotes() -> None:
    facts = [
        KeyFact(fact="f", title="Luna 25", quote="CRASHED  on the moon"),
        KeyFact(fact="f", title="Luna 25", quote="landed softly"),
        KeyFact(fact="f", title="Nope", quote="x"),
    ]
    case = GoldenCase(id="x", split="dev", question="q", key_facts=facts)
    docs = [Document(title="Luna 25", url="u", text="It crashed on the\nMoon.")]
    assert check_quotes([case], docs) == [
        "x fact 2: quote not found in 'Luna 25'",
        "x fact 3: unknown title 'Nope'",
    ]


def test_committed_golden_dataset_matches_the_committed_corpus() -> None:
    """Every key fact is really in the corpus, so a miss is the system's fault."""
    settings = DeepResearchSettings(_env_file=None)
    cases = load_golden(settings.evals_dir / "golden.jsonl")
    assert len(cases) >= 10
    assert {c.split for c in cases} == {"dev", "test"}
    assert check_quotes(cases, load_documents(settings.corpus_dir)) == []
