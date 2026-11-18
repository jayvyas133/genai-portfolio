"""Tests for the report eval: retrieval check, judges, scoring and the summary."""

import json
from datetime import UTC, datetime
from pathlib import Path

from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.errors import AuthenticationError
from aicore.models import LLMRequest, Message
from aicore.providers.fakeai import FakeProvider
from aicore.tracing import InMemoryTracer
from deep_research.evaluate import (
    EvalRun,
    evaluate,
    facts_retrieved,
    format_run,
    judge_facts,
)
from deep_research.golden import GoldenCase, KeyFact
from deep_research.report import Report, Researcher
from deep_research.search import Passage

SOURCES = [
    Passage(id="l#0", title="Luna 25", section="S", url="u", text="Luna 25 crashed on the Moon."),
    Passage(id="c#0", title="Chandrayaan-3", section="S", url="u", text="It landed in 2023."),
]
FACTS = [
    KeyFact(fact="Luna 25 crashed.", title="Luna 25", quote="crashed on the moon"),
    KeyFact(fact="Chandrayaan-3 landed in 2023.", title="Chandrayaan-3", quote="landed in 2023"),
    KeyFact(fact="Luna 25 launched on Soyuz.", title="Luna 25", quote="Soyuz-2 rocket"),
]
CASE = GoldenCase(id="dr-x", split="dev", question="Compare them.", key_facts=FACTS)


def _facts_reply(*stated: bool) -> str:
    verdicts = [{"fact": i, "reason": "r", "stated": s} for i, s in enumerate(stated, start=1)]
    return json.dumps({"verdicts": verdicts})


def _support_reply(supported: bool) -> str:
    return json.dumps({"reason": "checked", "supported": supported})


def _client(script: list[str | Exception], tracer: InMemoryTracer) -> LLMClient:
    return LLMClient(
        FakeProvider(script), LLMSettings(_env_file=None), tracer=tracer, base_delay_s=0
    )


def test_facts_retrieved_needs_the_quote_in_a_passage_of_the_right_document() -> None:
    wrong_doc = KeyFact(fact="f", title="Chandrayaan-3", quote="crashed on the moon")
    assert facts_retrieved([*FACTS, wrong_doc], SOURCES) == [True, True, False, False]


async def test_judge_facts_rejects_a_reply_with_the_wrong_number_of_verdicts() -> None:
    tracer = InMemoryTracer()
    client = _client([_facts_reply(True), "not json"], tracer)
    assert await judge_facts(FACTS, "report", client, "judge") is None
    assert await judge_facts(FACTS, "report", client, "judge") is None


def _researcher(client: LLMClient) -> Researcher:
    """A researcher that makes one model call and returns its text with ``SOURCES``."""

    async def research(question: str) -> Report:
        request = LLMRequest(messages=[Message(role="user", content=question)])
        response = await client.complete(request)
        return Report(question=question, text=response.text, sources=SOURCES)

    return research


REPORT = "Luna 25 crashed [1]. Chandrayaan-3 landed in 2023 [2]. It was close [7]. Wow."


async def test_evaluate_scores_a_report_and_splits_research_and_judge_calls() -> None:
    tracer = InMemoryTracer()
    client = _client(
        [REPORT, _facts_reply(True, True, False), _support_reply(True), _support_reply(False)],
        tracer,
    )
    results, _ = await evaluate([CASE], _researcher(client), client, tracer, judge_model="judge")

    (r,) = results
    assert r.error is None
    assert r.facts_stated == [True, True, False]
    assert r.facts_retrieved == [True, True, False]
    assert r.titles_found == ["Luna 25", "Chandrayaan-3"]
    assert (r.sentences, r.sentences_cited) == (4, 3)
    assert r.invalid_markers == [7]
    # The sentence citing the invalid [7] is not sent to the support judge.
    assert (r.claims_judged, r.claims_supported) == (2, 1)
    assert r.unsupported == ["Chandrayaan-3 landed in 2023 [2]. -- checked"]
    assert r.llm_calls == 1
    assert [rec.model for rec in tracer.records[1:]] == ["judge"] * 3


async def test_evaluate_caps_support_checks_and_survives_judge_errors() -> None:
    tracer = InMemoryTracer()
    client = _client([REPORT, "garbage", "garbage"], tracer)
    results, _ = await evaluate(
        [CASE], _researcher(client), client, tracer, judge_model="judge", max_claims=1
    )
    (r,) = results
    assert r.fact_judge_error
    assert r.facts_stated == []
    assert (r.claims_judged, r.support_judge_errors) == (0, 1)


async def test_evaluate_records_a_failing_researcher_and_moves_on() -> None:
    tracer = InMemoryTracer()
    error = AuthenticationError("bad key")
    client = _client([error], tracer)
    results, judge_cost = await evaluate(
        [CASE], _researcher(client), client, tracer, judge_model="judge"
    )
    (r,) = results
    assert r.error == "AuthenticationError"
    assert judge_cost == 0.0


async def test_format_run_shows_metrics_and_save_uses_a_fixed_name(tmp_path: Path) -> None:
    tracer = InMemoryTracer()
    client = _client(
        [REPORT, _facts_reply(True, True, False), _support_reply(True), _support_reply(True)],
        tracer,
    )
    results, judge_cost = await evaluate(
        [CASE], _researcher(client), client, tracer, judge_model="judge"
    )
    run = EvalRun(
        researcher="single-shot",
        model="m",
        judge_model="judge",
        writer_prompt_version="v1",
        judge_version="v1",
        split="dev",
        search_k=5,
        max_claims=8,
        created_at=datetime.now(UTC),
        cases=results,
        judge_cost_usd=judge_cost,
    )
    text = format_run(run)
    assert "| Key facts stated (judge) | 0.67 (2/3) |" in text
    assert "| Cited sentences supported (judge) | 1.00 (2/2) |" in text
    assert "invalid citation markers: 1" in text
    assert "- dr-x: facts stated 2/3, retrieved 2/3, supported 2/2" in text
    assert run.save(tmp_path).name == "single-shot-dev.json"
