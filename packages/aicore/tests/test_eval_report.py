"""Tests for the Markdown rendering of an ``EvalReport``."""

from datetime import UTC, datetime

from aicore.evals.models import CaseResult, EvalReport, Score
from aicore.evals.report import to_markdown


def make_report(results: list[CaseResult], cost: float | None = None) -> EvalReport:
    """Wrap ``results`` in a report with a fixed start time."""
    return EvalReport(
        name="injection",
        started_at=datetime(2026, 11, 16, 10, 30, tzinfo=UTC),
        results=results,
        llm_calls=2 if cost is not None else 0,
        total_cost_usd=cost,
    )


def passing(case_id: str) -> CaseResult:
    """Return a passing result."""
    score = Score(name="exact_match", value=1.0, passed=True)
    return CaseResult(case_id=case_id, scores=[score], latency_ms=2.0)


def failing(case_id: str, reason: str) -> CaseResult:
    """Return a result that failed ``exact_match`` with ``reason``."""
    score = Score(name="exact_match", value=0.0, passed=False, reason=reason)
    return CaseResult(case_id=case_id, scores=[score], latency_ms=4.0)


def test_summary_table() -> None:
    """The headline numbers appear in the metric table."""
    text = to_markdown(make_report([passing("c1"), failing("c2", "wrong")], cost=0.00005))
    assert "### Eval: injection" in text
    assert "2026-11-16T10:30+00:00" in text
    assert "| Cases | 2 |" in text
    assert "| Passed | 1 (50.0%) |" in text
    assert "| Target errors | 0 |" in text
    assert "| Latency p50 / p95 | 2.00 ms / 4.00 ms |" in text
    assert "| LLM calls | 2 |" in text
    assert "| Total cost | USD 0.000050 |" in text
    assert "| exact_match | 50.0% |" in text


def test_unknown_cost_is_labelled() -> None:
    """``None`` cost is shown as unknown, never as zero."""
    assert "| Total cost | unknown |" in to_markdown(make_report([passing("c1")]))


def test_all_passed_has_no_failure_table() -> None:
    """A clean run does not print an empty failure section."""
    assert "Failed cases" not in to_markdown(make_report([passing("c1")]))


def test_failures_show_reason_and_error() -> None:
    """Each failed case says which scorer failed and why, or which error occurred."""
    errored = CaseResult(case_id="c3", error="ProviderTimeoutError", latency_ms=1.0)
    text = to_markdown(make_report([failing("c2", "expected true, got false"), errored]))
    assert "Failed cases (2 of 2 shown):" in text
    assert "| c2 | exact_match: expected true, got false |" in text
    assert "| c3 | error: ProviderTimeoutError |" in text


def test_failures_are_capped() -> None:
    """A bad run lists only the first ``max_failures`` failed cases."""
    report = make_report([failing(f"c{i}", "wrong") for i in range(5)])
    text = to_markdown(report, max_failures=2)
    assert "Failed cases (2 of 5 shown):" in text
    assert "| c1 |" in text
    assert "| c2 |" not in text


def test_pipes_and_newlines_cannot_break_the_table() -> None:
    """Reasons copied from model output are escaped for Markdown."""
    text = to_markdown(make_report([failing("c1", "got a|b\nnext")]))
    assert "| c1 | exact_match: got a\\|b next |" in text
