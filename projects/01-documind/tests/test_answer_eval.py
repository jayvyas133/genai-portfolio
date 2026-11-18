"""Tests for the answer eval: citation scoring, the judge, the loop and the report."""

import json
from datetime import UTC, datetime

from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.providers.fakeai import FakeProvider
from aicore.tracing import InMemoryTracer
from documind.answer_eval import (
    AnswerEvalRun,
    build_judge_request,
    evaluate_answers,
    format_answer_run,
    judge_answer,
    score_citations,
)
from documind.generation import Answer, Citation
from documind.golden import Evidence, GoldenCase
from documind.store import RetrievedChunk

RELEASE = RetrievedChunk(
    chunk_id=11,
    document_title="CUDA",
    section="History",
    text="CUDA was initially released in 2007 by Nvidia.",
    score=0.9,
)
OTHER = RetrievedChunk(
    chunk_id=22, document_title="Nvidia", section="Products", text="Makes GPUs.", score=0.5
)
FACTUAL = GoldenCase(
    id="cuda-001",
    question="When was CUDA first released?",
    reference_answer="In 2007.",
    evidence=[Evidence(title="CUDA", quote="initially released in 2007")],
    kind="factual",
    split="dev",
)
UNANSWERABLE = GoldenCase(
    id="none-001",
    question="What is the capital of France?",
    reference_answer="Not in the documents.",
    kind="unanswerable",
    split="dev",
)


class FixedRetriever:
    """Returns the same chunks for every query."""

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return ``RELEASE`` then ``OTHER``."""
        return [RELEASE, OTHER][:k]


def cite(*numbers: int) -> list[Citation]:
    """Citations for prompt ``numbers`` over ``[RELEASE, OTHER]``."""
    chunks = [RELEASE, OTHER]
    return [
        Citation(
            number=n,
            chunk_id=chunks[n - 1].chunk_id,
            document_title=chunks[n - 1].document_title,
            section=chunks[n - 1].section,
        )
        for n in numbers
    ]


def test_score_citations_flags_chunks_without_evidence() -> None:
    """[1] holds the quote, [2] does not; the one evidence passage is covered."""
    answer = Answer(text="2007 [1][2].", citations=cite(1, 2), chunks=[RELEASE, OTHER])
    assert score_citations(FACTUAL, answer) == ([True, False], 1)


def test_score_citations_of_a_refusal_is_empty() -> None:
    """No citations: nothing supported, nothing covered."""
    answer = Answer(text="", refusal_reason="model", chunks=[RELEASE])
    assert score_citations(FACTUAL, answer) == ([], 0)


def test_judge_request_strips_markers_and_includes_the_reference() -> None:
    """The judge grades facts, so source markers are removed before it sees the answer."""
    request = build_judge_request(FACTUAL, "It was released in 2007 [1][2].")
    user = request.messages[-1].content
    assert "Reference answer: In 2007." in user
    assert user.endswith("Answer to grade: It was released in 2007.")
    assert request.temperature == 0.0


async def test_invalid_judge_reply_is_none() -> None:
    """A judge reply that is not valid JSON is reported, not guessed."""
    client = LLMClient(FakeProvider(["looks right to me"]), LLMSettings(_env_file=None))
    assert await judge_answer(FACTUAL, "2007 [1].", client) is None


async def test_evaluate_answers_judges_only_answered_answerable_cases() -> None:
    """Two answer calls and one judge call: the refused unanswerable case is not judged."""
    fake = FakeProvider(
        [
            json.dumps({"answer": "Released in 2007 [1][2].", "refused": False}),
            json.dumps({"reason": "Matches the reference year.", "verdict": "correct"}),
            json.dumps({"answer": "", "refused": True}),
        ]
    )
    tracer = InMemoryTracer()
    client = LLMClient(fake, LLMSettings(_env_file=None), tracer=tracer, base_delay_s=0)

    report = await evaluate_answers([FACTUAL, UNANSWERABLE], FixedRetriever(), client, tracer)

    factual, unanswerable = report.cases
    assert factual.verdict == "correct"
    assert factual.citation_supported == [True, False]
    assert (factual.evidence_cited, factual.evidence_total) == (1, 1)
    assert factual.cited_chunk_ids == [11, 22]
    assert unanswerable.refusal_reason == "model"
    assert unanswerable.verdict is None
    assert len(fake.calls) == 3


async def test_report_shows_metrics_refusals_and_verdicts() -> None:
    """The rendered report has the summary rows and lists each verdict for spot-checking."""
    fake = FakeProvider(
        [
            json.dumps({"answer": "Released in 2007 [1][2].", "refused": False}),
            json.dumps({"reason": "Matches the reference year.", "verdict": "correct"}),
            json.dumps({"answer": "", "refused": True}),
        ]
    )
    tracer = InMemoryTracer()
    client = LLMClient(fake, LLMSettings(_env_file=None), tracer=tracer, base_delay_s=0)
    report = await evaluate_answers([FACTUAL, UNANSWERABLE], FixedRetriever(), client, tracer)
    run = AnswerEvalRun(
        prompt_version="v2",
        judge_version="v1",
        model="m",
        retriever="hybrid-rerank",
        split="dev",
        context_chunks=5,
        created_at=datetime.now(UTC),
        report=report,
    )

    text = format_answer_run(run)

    assert "| Citation precision | 0.50 (1/2) | 0.50 (1/2) | - |" in text
    assert "| Correct end to end (of all) | 1.00 (1/1) | 1.00 (1/1) | - |" in text
    assert "Unanswerable refused: 1.00 (1/1)" in text
    assert "False refusals: 0.00 (0/1) (none)" in text
    assert "- cuda-001 [correct] Matches the reference year." in text
