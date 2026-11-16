"""Tests for precision and recall of yes/no detectors."""

from datetime import UTC, datetime

import pytest
from aicore.evals.metrics import BinaryMetrics, binary_metrics
from aicore.evals.models import CaseResult, EvalCase, EvalReport
from pydantic import JsonValue


def make_case(case_id: str, expected: JsonValue) -> EvalCase:
    """Return a labelled case."""
    return EvalCase(id=case_id, input="text", expected=expected)


def make_result(case_id: str, output: JsonValue, error: str | None = None) -> CaseResult:
    """Return a detector result with the given prediction."""
    return CaseResult(case_id=case_id, output=output, error=error, latency_ms=1.0)


def make_report(results: list[CaseResult]) -> EvalReport:
    """Wrap ``results`` in a report."""
    return EvalReport(
        name="detector", started_at=datetime(2026, 11, 16, tzinfo=UTC), results=results
    )


def test_precision_and_recall() -> None:
    """3 caught, 1 missed, 1 false alarm, 2 correctly ignored."""
    metrics = BinaryMetrics(tp=3, fn=1, fp=1, tn=2)
    assert metrics.precision == 0.75
    assert metrics.recall == 0.75


def test_rates_are_zero_when_undefined() -> None:
    """Nothing flagged and nothing to find: no division by zero."""
    metrics = BinaryMetrics(tp=0, fp=0, fn=0, tn=5)
    assert (metrics.precision, metrics.recall) == (0.0, 0.0)


def test_counts_each_quadrant() -> None:
    """Each (expected, predicted) pair lands in the right count."""
    cases = [make_case("tp", True), make_case("fn", True), make_case("fp", False)]
    cases.append(make_case("tn", False))
    report = make_report(
        [
            make_result("tp", True),
            make_result("fn", False),
            make_result("fp", True),
            make_result("tn", False),
        ]
    )
    assert binary_metrics(cases, report) == BinaryMetrics(tp=1, fn=1, fp=1, tn=1)


def test_errors_and_bad_outputs_count_as_wrong() -> None:
    """A crash or a non-bool output never earns credit."""
    cases = [make_case("a", True), make_case("b", False), make_case("c", True)]
    report = make_report(
        [
            make_result("a", None, error="ProviderTimeoutError"),  # missed attack
            make_result("b", None, error="ProviderTimeoutError"),  # false alarm
            make_result("c", "yes"),  # not a bool: missed attack
        ]
    )
    assert binary_metrics(cases, report) == BinaryMetrics(tp=0, fn=2, fp=1, tn=0)


def test_rejects_non_bool_expected() -> None:
    """Labels must be true/false; anything else is a dataset bug."""
    with pytest.raises(ValueError, match="expected must be true or false"):
        binary_metrics([make_case("a", "yes")], make_report([]))


def test_rejects_result_without_case() -> None:
    """A result for an unknown case means the wrong dataset was passed."""
    with pytest.raises(ValueError, match="no matching case"):
        binary_metrics([make_case("a", True)], make_report([make_result("z", True)]))
