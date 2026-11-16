"""Precision and recall for yes/no detectors such as guardrails.

Pass rate hides how a detector is wrong. Blocking a normal user (a
false positive) and letting an attack through (a false negative) cost
different things, so they are counted separately.
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from aicore.evals.models import EvalCase, EvalReport


class BinaryMetrics(BaseModel):
    """Confusion-matrix counts for a yes/no detector, plus derived rates.

    *Positive* means the case should be flagged (``expected`` is true):
    ``tp`` caught, ``fn`` missed, ``fp`` wrongly flagged, ``tn``
    correctly left alone.
    """

    model_config = ConfigDict(frozen=True)

    tp: int = Field(ge=0)
    fp: int = Field(ge=0)
    fn: int = Field(ge=0)
    tn: int = Field(ge=0)

    @property
    def precision(self) -> float:
        """Of the cases flagged, the fraction that should have been (0.0 if none)."""
        flagged = self.tp + self.fp
        return self.tp / flagged if flagged else 0.0

    @property
    def recall(self) -> float:
        """Of the cases that should be flagged, the fraction that were (0.0 if none)."""
        positives = self.tp + self.fn
        return self.tp / positives if positives else 0.0


def binary_metrics(cases: Sequence[EvalCase], report: EvalReport) -> BinaryMetrics:
    """Count a detector's hits and misses from an eval report.

    The target's output must be ``True`` (flagged) or ``False``. Results
    are matched to cases by id. A case whose target raised, or whose
    output is not a bool, counts as a wrong prediction, so a broken
    detector can never look better than it is.

    Raises:
        ValueError: If a case's ``expected`` is not a bool, or a result
            has no matching case.
    """
    expected_by_id: dict[str, bool] = {}
    for case in cases:
        if not isinstance(case.expected, bool):
            raise ValueError(f"case {case.id}: expected must be true or false")
        expected_by_id[case.id] = case.expected

    tp = fp = fn = tn = 0
    for result in report.results:
        if result.case_id not in expected_by_id:
            raise ValueError(f"result {result.case_id} has no matching case")
        expected = expected_by_id[result.case_id]
        predicted = result.output
        if result.error is not None or not isinstance(predicted, bool):
            predicted = not expected  # a missing or malformed prediction is always wrong
        if expected and predicted:
            tp += 1
        elif expected:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    return BinaryMetrics(tp=tp, fp=fp, fn=fn, tn=tn)
