"""Tests for retrieval metrics, checked against hand-computed values."""

from dataclasses import dataclass

import pytest
from documind.chunking import Chunk
from documind.golden import Evidence, GoldenCase, Kind, Source
from documind.metrics import CaseRetrieval, RetrievalReport, is_relevant, score_case
from pydantic import ValidationError


@dataclass(frozen=True)
class Hit:
    """A minimal ranked chunk: just a title and text."""

    document_title: str
    text: str


CUDA = Evidence(title="CUDA", quote="released in 2007")
HBM = Evidence(title="HBM", quote="stacks DRAM dies")


def golden(*evidence: Evidence, kind: Kind = "factual") -> GoldenCase:
    """A dev case with the given evidence."""
    return GoldenCase(
        id="c",
        question="q",
        reference_answer="a",
        evidence=list(evidence),
        kind=kind,
        split="dev",
    )


def report(*ranks: list[int | None]) -> RetrievalReport:
    """A report over cases with the given evidence ranks."""
    return RetrievalReport(
        cases=[
            CaseRetrieval(case_id=str(i), kind="factual", retrieved=10, evidence_ranks=r)
            for i, r in enumerate(ranks)
        ]
    )


def test_relevance_needs_same_document_and_quote() -> None:
    """Title must match exactly; the quote ignores case and whitespace."""
    assert is_relevant(CUDA, Hit("CUDA", "It was Released\nin 2007."))
    assert not is_relevant(CUDA, Hit("OpenCL", "released in 2007"))
    assert not is_relevant(CUDA, Hit("CUDA", "released in 2008"))


def test_real_chunks_satisfy_the_protocol() -> None:
    """``Chunk`` from the chunker can be scored directly."""
    chunk = Chunk(document_title="CUDA", ordinal=0, section="History", text="released in 2007")

    assert is_relevant(CUDA, chunk)


def test_score_case_records_first_rank_per_evidence() -> None:
    """Each passage gets the rank of its first match; missing ones get None."""
    ranked = [
        Hit("Other", "noise"),
        Hit("CUDA", "CUDA was released in 2007"),
        Hit("CUDA", "also released in 2007"),
    ]

    scored = score_case(golden(CUDA, HBM, kind="multi-hop"), ranked)

    assert scored.evidence_ranks == [2, None]
    assert scored.first_rank == 2
    assert scored.retrieved == 3


def test_score_case_rejects_unanswerable() -> None:
    """Unanswerable cases are scored on refusal, not retrieval."""
    with pytest.raises(ValueError, match="unanswerable"):
        score_case(golden(kind="unanswerable"), [])


def test_case_metrics_at_k() -> None:
    """Multi-hop with passages at ranks 2 and 5: hit from k=2, full recall at k=5."""
    case = CaseRetrieval(case_id="c", kind="multi-hop", retrieved=10, evidence_ranks=[5, 2])

    assert (case.hit(1), case.hit(2)) == (False, True)
    assert (case.recall(1), case.recall(2), case.recall(5)) == (0.0, 0.5, 1.0)
    assert case.reciprocal_rank() == 0.5


def test_missed_case_scores_zero() -> None:
    """Nothing retrieved: no hit, no recall, reciprocal rank 0."""
    case = CaseRetrieval(case_id="c", kind="factual", retrieved=10, evidence_ranks=[None])

    assert (case.hit(10), case.recall(10), case.reciprocal_rank()) == (False, 0.0, 0.0)


def test_report_averages_over_cases() -> None:
    """Ranks 1, 3 and a miss: MRR = (1 + 1/3 + 0) / 3; hits@3 = 2 of 3."""
    r = report([1], [3], [None])

    assert r.hits(1) == 1
    assert r.hits(3) == 2
    assert r.hit_rate(3) == pytest.approx(2 / 3)
    assert r.recall(3) == pytest.approx(2 / 3)
    assert r.mrr() == pytest.approx((1 + 1 / 3) / 3)


def test_recall_differs_from_hit_rate_on_multi_hop() -> None:
    """One passage of two found: a hit, but only half recall."""
    r = report([1, None])

    assert r.hit_rate(5) == 1.0
    assert r.recall(5) == 0.5


def test_empty_report_is_rejected() -> None:
    """A report over zero cases would divide by zero; fail loudly instead."""
    with pytest.raises(ValidationError):
        RetrievalReport(cases=[])


def test_report_round_trips_through_json() -> None:
    """Saved results load back unchanged, so M4 can compare runs."""
    r = report([2, None], [1])

    assert RetrievalReport.model_validate_json(r.model_dump_json()) == r


def test_alternative_sources_also_count() -> None:
    """A fact stated in a second document is relevant there too."""
    evidence = Evidence(
        title="Radeon",
        quote="for US$5.4 billion",
        alternatives=[Source(title="AMD", quote="approximately $5.4 billion")],
    )

    assert is_relevant(evidence, Hit("AMD", "paid approximately $5.4 billion"))
    assert is_relevant(evidence, Hit("Radeon", "bought for US$5.4 billion"))
    assert not is_relevant(evidence, Hit("AMD", "for US$5.4 billion"))


def test_only_filters_by_kind() -> None:
    """``only`` keeps one kind's cases, or returns None when there are none."""
    r = RetrievalReport(
        cases=[
            CaseRetrieval(case_id="a", kind="factual", retrieved=1, evidence_ranks=[1]),
            CaseRetrieval(case_id="b", kind="multi-hop", retrieved=1, evidence_ranks=[None, 1]),
        ]
    )

    factual = r.only("factual")
    assert factual is not None
    assert [c.case_id for c in factual.cases] == ["a"]
    assert r.only("unanswerable") is None


def test_latency_percentile_uses_nearest_rank() -> None:
    """p50 of 10..100 ms is 50; p95 is 100; unmeasured cases are ignored."""
    latencies: list[float | None] = [float(x) for x in range(10, 101, 10)]
    latencies.append(None)
    cases = [
        CaseRetrieval(
            case_id=str(i), kind="factual", retrieved=1, evidence_ranks=[1], latency_ms=ms
        )
        for i, ms in enumerate(latencies)
    ]
    r = RetrievalReport(cases=cases)

    assert (r.latency_percentile(50), r.latency_percentile(95)) == (50.0, 100.0)
    assert report([1]).latency_percentile(50) is None
    with pytest.raises(ValueError, match="pct"):
        r.latency_percentile(0)
