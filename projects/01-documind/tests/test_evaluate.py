"""Tests for the retrieval eval runner, result files and the Markdown table."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from documind.evaluate import EvalRun, evaluate_retrieval, format_run
from documind.golden import Evidence, GoldenCase
from documind.metrics import CaseRetrieval, RetrievalReport
from documind.store import RetrievedChunk


def golden(case_id: str, *quotes: str) -> GoldenCase:
    """A dev case whose evidence is ``quotes`` in document "D" (none: unanswerable)."""
    kind = "unanswerable" if not quotes else "factual" if len(quotes) == 1 else "multi-hop"
    return GoldenCase(
        id=case_id,
        question=f"question {case_id}",
        reference_answer="a",
        evidence=[Evidence(title="D", quote=q) for q in quotes],
        kind=kind,
        split="dev",
    )


def chunk(text: str) -> RetrievedChunk:
    """A retrieved chunk of document "D"."""
    return RetrievedChunk(chunk_id=0, document_title="D", section="S", text=text, score=0.5)


class ScriptedRetriever:
    """Returns a fixed ranking per question and records every call."""

    def __init__(self, rankings: dict[str, list[RetrievedChunk]]) -> None:
        """``rankings`` maps a question to the chunks to return for it."""
        self.rankings = rankings
        self.calls: list[tuple[str, int]] = []

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Record the call and return the scripted chunks, cut to ``k``."""
        self.calls.append((query, k))
        return self.rankings.get(query, [])[:k]


def run_of(report: RetrievalReport) -> EvalRun:
    """An ``EvalRun`` wrapping ``report`` with fixed metadata."""
    return EvalRun(
        retriever="dense",
        split="dev",
        k=10,
        embed_model="m",
        chunk_words=200,
        chunk_overlap_words=40,
        created_at=datetime(2026, 11, 17, tzinfo=UTC),
        skipped_unanswerable=1,
        report=report,
    )


async def test_evaluate_scores_answerable_cases_and_skips_the_rest() -> None:
    """Unanswerable cases are counted, not retrieved; ranks and latency are recorded."""
    retriever = ScriptedRetriever(
        {
            "question a": [chunk("noise"), chunk("the answer")],
            "question b": [chunk("first fact"), chunk("second fact")],
        }
    )
    cases = [golden("a", "answer"), golden("b", "first", "second"), golden("c")]

    report, skipped = await evaluate_retrieval(cases, retriever, k=5)

    assert skipped == 1
    assert retriever.calls == [("question a", 5), ("question b", 5)]
    assert [c.evidence_ranks for c in report.cases] == [[2], [1, 2]]
    assert all(c.latency_ms is not None and c.latency_ms >= 0 for c in report.cases)


async def test_evaluate_needs_an_answerable_case() -> None:
    """Only unanswerable cases means nothing to score: fail loudly."""
    with pytest.raises(ValueError, match="no answerable cases"):
        await evaluate_retrieval([golden("c")], ScriptedRetriever({}), k=5)


def test_format_run_shows_counts_kinds_and_latency() -> None:
    """The table has raw counts, a column per kind and only cut-offs <= k."""
    report = RetrievalReport(
        cases=[
            CaseRetrieval(
                case_id="a", kind="factual", retrieved=10, evidence_ranks=[1], latency_ms=100
            ),
            CaseRetrieval(
                case_id="b", kind="factual", retrieved=10, evidence_ranks=[None], latency_ms=300
            ),
        ]
    )

    text = format_run(run_of(report), ks=(1, 20))

    assert "2 cases (1 unanswerable skipped)" in text
    assert "| Metric | All (2) | Factual (2) | Multi-hop (0) |" in text
    assert "| Hit rate@1 | 0.50 (1/2) | 0.50 (1/2) | - |" in text
    assert "@20" not in text
    assert "| MRR@10 | 0.50 | 0.50 | - |" in text
    assert "p50 100 ms, p95 300 ms" in text


def test_save_overwrites_a_fixed_name_and_round_trips(tmp_path: Path) -> None:
    """One file per retriever and split, so git diffs show what changed."""
    run = run_of(
        RetrievalReport(
            cases=[CaseRetrieval(case_id="a", kind="factual", retrieved=10, evidence_ranks=[3])]
        )
    )

    first = run.save(tmp_path / "results")
    second = run.save(tmp_path / "results")

    assert first == second == tmp_path / "results" / "dense-dev.json"
    assert EvalRun.model_validate_json(first.read_text(encoding="utf-8")) == run
