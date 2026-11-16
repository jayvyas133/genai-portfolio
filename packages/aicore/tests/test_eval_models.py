"""Tests for the evaluation data models."""

from datetime import UTC, datetime

import pytest
from aicore.evals.models import CaseResult, EvalCase, EvalReport, Score
from pydantic import ValidationError


def make_score(passed: bool) -> Score:
    """Return a score with the given verdict."""
    return Score(name="s", value=1.0 if passed else 0.0, passed=passed)


def test_case_defaults() -> None:
    """Only id and input are required."""
    case = EvalCase(id="c1", input="hello")
    assert case.expected is None
    assert case.tags == []


def test_case_accepts_dict_expected() -> None:
    """Structured expectations (extraction tasks) are allowed."""
    case = EvalCase(id="c1", input="I live in Pune", expected={"city": "Pune"})
    assert case.expected == {"city": "Pune"}


def test_case_rejects_unknown_field() -> None:
    """A typo in a dataset line must fail, not silently drop data."""
    with pytest.raises(ValidationError):
        EvalCase.model_validate({"id": "c1", "input": "x", "expcted": "y"})


def test_case_rejects_empty_id() -> None:
    """Every case needs an id so results can be traced back to it."""
    with pytest.raises(ValidationError):
        EvalCase(id="", input="x")


def test_score_value_must_be_between_0_and_1() -> None:
    """Scores are normalised so different scorers can be compared."""
    with pytest.raises(ValidationError):
        Score(name="s", value=1.5, passed=True)


def test_result_passes_when_all_scores_pass() -> None:
    """No error and every scorer passed means the case passed."""
    result = CaseResult(
        case_id="c1",
        output="Pune",
        scores=[make_score(True), make_score(True)],
        latency_ms=1.0,
    )
    assert result.passed


def test_result_fails_when_one_score_fails() -> None:
    """A single failing scorer fails the whole case."""
    result = CaseResult(
        case_id="c1",
        output="Pune",
        scores=[make_score(True), make_score(False)],
        latency_ms=1.0,
    )
    assert not result.passed


def test_result_fails_on_error() -> None:
    """A target that raised never counts as passed."""
    result = CaseResult(case_id="c1", error="ProviderTimeoutError", latency_ms=1.0)
    assert not result.passed


# --- EvalReport summary -----------------------------------------------------


def make_report(results: list[CaseResult]) -> EvalReport:
    """Wrap ``results`` in a report."""
    return EvalReport(name="r", started_at=datetime(2026, 11, 16, tzinfo=UTC), results=results)


def ok(case_id: str, latency_ms: float = 1.0) -> CaseResult:
    """Return a passing result with the given latency."""
    return CaseResult(case_id=case_id, scores=[make_score(True)], latency_ms=latency_ms)


def test_pass_rate_and_error_count() -> None:
    """Errors count as failures in the overall pass rate."""
    report = make_report(
        [
            ok("c1"),
            ok("c2"),
            CaseResult(case_id="c3", scores=[make_score(False)], latency_ms=1.0),
            CaseResult(case_id="c4", error="ProviderTimeoutError", latency_ms=1.0),
        ]
    )
    assert report.pass_rate == 0.5
    assert report.error_count == 1


def test_empty_report_summary_is_zero() -> None:
    """An empty run does not divide by zero."""
    report = make_report([])
    assert report.pass_rate == 0.0
    assert report.scorer_pass_rates() == {}
    assert report.latency_percentile(95) == 0.0


def test_scorer_pass_rates_count_errors_as_fails() -> None:
    """A case that errored lowers every scorer's rate instead of being skipped."""
    a_pass = Score(name="a", value=1.0, passed=True)
    b_pass = Score(name="b", value=1.0, passed=True)
    b_fail = Score(name="b", value=0.0, passed=False)
    report = make_report(
        [
            CaseResult(case_id="c1", scores=[a_pass, b_fail], latency_ms=1.0),
            CaseResult(case_id="c2", scores=[a_pass, b_pass], latency_ms=1.0),
            CaseResult(case_id="c3", error="ProviderTimeoutError", latency_ms=1.0),
            CaseResult(case_id="c4", scores=[a_pass, b_fail], latency_ms=1.0),
        ]
    )
    assert report.scorer_pass_rates() == {"a": 0.75, "b": 0.25}


def test_latency_percentile_nearest_rank() -> None:
    """p50 and p95 are real observed values from the sorted latencies."""
    latencies = [5000.0, *[float(ms) for ms in range(100, 2000, 100)]]  # 20 values
    report = make_report([ok(f"c{i}", ms) for i, ms in enumerate(latencies)])
    assert report.latency_percentile(50) == 1000.0  # 10th of 20
    assert report.latency_percentile(95) == 1900.0  # 19th of 20
    assert report.latency_percentile(100) == 5000.0  # the slowest


@pytest.mark.parametrize("p", [0, -1, 101])
def test_latency_percentile_rejects_bad_p(p: float) -> None:
    """Percentiles outside (0, 100] are a programming error."""
    with pytest.raises(ValueError, match="percentile"):
        make_report([ok("c1")]).latency_percentile(p)
