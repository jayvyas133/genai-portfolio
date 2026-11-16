"""Tests for ``run_eval``: ordering, concurrency, errors and scoring."""

import asyncio
import logging

import pytest
from aicore.errors import ProviderTimeoutError
from aicore.evals.models import EvalCase, Score
from aicore.evals.runner import run_eval
from aicore.evals.scorers import ExactMatch
from aicore.tracing import CallRecord, InMemoryTracer
from pydantic import JsonValue


def make_cases(n: int) -> list[EvalCase]:
    """Return ``n`` cases whose expected output is their own input."""
    return [EvalCase(id=f"c{i}", input=str(i), expected=str(i), tags=["t"]) for i in range(n)]


async def echo(case: EvalCase) -> JsonValue:
    """Target that returns the input, so every ``ExactMatch`` passes."""
    return case.input


class BrokenScorer:
    """Scorer with a bug: it always raises."""

    name = "broken"

    def score(self, case: EvalCase, output: JsonValue) -> Score:
        """Raise, as a buggy custom scorer would."""
        raise KeyError("oops")


async def test_scores_every_case() -> None:
    """Each case gets one result with one score per scorer."""
    report = await run_eval("demo", make_cases(3), echo, [ExactMatch()])
    assert report.name == "demo"
    assert len(report.results) == 3
    assert all(r.passed for r in report.results)
    assert all(len(r.scores) == 1 for r in report.results)
    assert report.results[0].output == "0"
    assert report.results[0].tags == ["t"]


async def test_results_keep_dataset_order() -> None:
    """Cases finishing out of order still come back in dataset order."""

    async def slow_first(case: EvalCase) -> JsonValue:
        """Make earlier cases slower than later ones."""
        await asyncio.sleep(0.01 * (5 - int(case.input)))
        return case.input

    report = await run_eval("order", make_cases(5), slow_first, [], concurrency=5)
    assert [r.case_id for r in report.results] == ["c0", "c1", "c2", "c3", "c4"]


async def test_concurrency_is_limited() -> None:
    """Never more than ``concurrency`` targets run at the same time."""
    running = 0
    peak = 0

    async def tracked(case: EvalCase) -> JsonValue:
        """Record how many calls are in flight."""
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return case.input

    await run_eval("limit", make_cases(10), tracked, [], concurrency=3)
    assert peak == 3


async def test_target_error_becomes_failed_result(caplog: pytest.LogCaptureFixture) -> None:
    """A failing case is recorded and the other cases still run."""

    async def flaky(case: EvalCase) -> JsonValue:
        """Fail on case c1 only."""
        if case.id == "c1":
            raise ProviderTimeoutError("secret prompt text")
        return case.input

    with caplog.at_level(logging.WARNING):
        report = await run_eval("errors", make_cases(3), flaky, [ExactMatch()])

    failed = report.results[1]
    assert failed.error == "ProviderTimeoutError"
    assert failed.output is None
    assert failed.scores == []
    assert not failed.passed
    assert report.results[0].passed and report.results[2].passed
    assert "c1" in caplog.text
    assert "secret prompt text" not in caplog.text


async def test_scorer_error_is_not_swallowed() -> None:
    """A buggy scorer stops the run instead of looking like a bad model."""
    with pytest.raises(KeyError):
        await run_eval("bug", make_cases(1), echo, [BrokenScorer()])


async def test_latency_is_measured() -> None:
    """Latency covers the time spent in the target."""

    async def sleepy(case: EvalCase) -> JsonValue:
        """Take about 20 ms."""
        await asyncio.sleep(0.02)
        return case.input

    report = await run_eval("latency", make_cases(1), sleepy, [])
    assert report.results[0].latency_ms >= 15


async def test_empty_dataset_gives_empty_report() -> None:
    """No cases, no results (``load_dataset`` already rejects empty files)."""
    report = await run_eval("empty", [], echo, [ExactMatch()])
    assert report.results == []


def make_record(cost: float | None) -> CallRecord:
    """Return a call record with the given cost."""
    return CallRecord(
        provider="fake", model="m", latency_ms=1.0, attempts=1, outcome="ok", cost_usd=cost
    )


async def test_tracer_gives_call_count_and_total_cost() -> None:
    """Only records added during the run are counted; unknown costs are skipped."""
    tracer = InMemoryTracer()
    tracer.record(make_record(5.0))  # from earlier work, must not be counted

    async def calls_llm(case: EvalCase) -> JsonValue:
        """Pretend to make one priced and one unpriced LLM call."""
        tracer.record(make_record(0.25))
        tracer.record(make_record(None))
        return case.input

    report = await run_eval("cost", make_cases(2), calls_llm, [], tracer=tracer)
    assert report.llm_calls == 4
    assert report.total_cost_usd == pytest.approx(0.5)


async def test_cost_is_none_without_priced_calls() -> None:
    """No tracer, or no known cost, means cost is unknown rather than 0."""
    report = await run_eval("no tracer", make_cases(1), echo, [])
    assert (report.llm_calls, report.total_cost_usd) == (0, None)

    tracer = InMemoryTracer()
    report = await run_eval("no calls", make_cases(1), echo, [], tracer=tracer)
    assert (report.llm_calls, report.total_cost_usd) == (0, None)


async def test_cache_hits_cost_zero_not_unknown() -> None:
    """A run served from the cache costs exactly 0.0."""
    tracer = InMemoryTracer()

    async def cached(case: EvalCase) -> JsonValue:
        """Pretend the LLM reply came from the cache."""
        tracer.record(make_record(0.0))
        return case.input

    report = await run_eval("cached", make_cases(3), cached, [], tracer=tracer)
    assert report.total_cost_usd == 0.0


async def test_concurrency_must_be_positive() -> None:
    """Zero concurrency would hang forever, so it is rejected."""
    with pytest.raises(ValueError, match="at least 1"):
        await run_eval("bad", make_cases(1), echo, [], concurrency=0)
