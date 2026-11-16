"""Data models for evaluation runs.

A run takes a list of ``EvalCase`` objects, produces one ``CaseResult``
per case and collects them in an ``EvalReport``. All models are frozen:
results describe something that already happened.
"""

import math
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class EvalCase(BaseModel):
    """One golden example: an input and what a good output looks like.

    ``expected`` can be any JSON value (a label, a string or a dict),
    so one model serves classification, extraction and Q&A datasets.
    Unknown fields are rejected so a typo in a hand-written dataset
    line fails loudly instead of leaving ``expected`` empty.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    input: str
    expected: JsonValue = None
    tags: list[str] = Field(default_factory=list)


class Score(BaseModel):
    """One scorer's verdict on one case.

    ``value`` is a grade from 0 to 1; ``passed`` is the scorer's own
    pass/fail decision, so fuzzy scorers can pass below 1.0.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    value: float = Field(ge=0.0, le=1.0)
    passed: bool
    reason: str = ""


class CaseResult(BaseModel):
    """What happened when one case was run and scored.

    If the target raised, ``error`` holds the exception class name,
    ``output`` is ``None`` and the case counts as failed.
    """

    model_config = ConfigDict(frozen=True)

    case_id: str
    output: JsonValue = None
    error: str | None = None
    scores: list[Score] = Field(default_factory=list)
    latency_ms: float = Field(ge=0.0)
    tags: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if the target succeeded and every scorer passed."""
        return self.error is None and all(s.passed for s in self.scores)


class EvalReport(BaseModel):
    """All results of one evaluation run over one dataset.

    Summary numbers are computed from ``results`` on demand, not
    stored, so they can never disagree with the results.
    ``llm_calls`` and ``total_cost_usd`` come from the tracer and keep
    their defaults when ``run_eval`` was given no tracer.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    started_at: datetime
    results: list[CaseResult]
    llm_calls: int = Field(default=0, ge=0)
    total_cost_usd: float | None = Field(default=None, ge=0.0)

    @property
    def pass_rate(self) -> float:
        """Fraction of cases that passed (0.0 for an empty run)."""
        if not self.results:
            return 0.0
        return sum(r.passed for r in self.results) / len(self.results)

    @property
    def error_count(self) -> int:
        """Number of cases where the target raised."""
        return sum(r.error is not None for r in self.results)

    def scorer_pass_rates(self) -> dict[str, float]:
        """Return each scorer's pass rate over all cases, in first-seen order.

        A case where the target raised has no scores and counts as a
        fail for every scorer, so errors cannot inflate a scorer's rate.
        """
        passes: dict[str, int] = {}
        for result in self.results:
            for score in result.scores:
                passes[score.name] = passes.get(score.name, 0) + int(score.passed)
        return {name: count / len(self.results) for name, count in passes.items()}

    def latency_percentile(self, p: float) -> float:
        """Return the ``p``-th percentile latency in ms (nearest-rank method).

        Nearest rank always returns a latency that was actually observed.
        Returns 0.0 for an empty run.

        Raises:
            ValueError: If ``p`` is not in the range (0, 100].
        """
        if not 0 < p <= 100:
            raise ValueError(f"percentile must be in (0, 100], got {p}")
        if not self.results:
            return 0.0
        latencies = sorted(r.latency_ms for r in self.results)
        rank = math.ceil(p / 100 * len(latencies))
        return latencies[rank - 1]
