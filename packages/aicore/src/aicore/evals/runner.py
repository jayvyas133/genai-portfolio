"""Run a target over a dataset and score every output.

A *target* is any async function ``(case) -> output``: a bare
``LLMClient`` call, a full RAG (Retrieval-Augmented Generation)
pipeline or a plain Python function such as a guardrail check.
The runner does not care what happens inside it.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime

from pydantic import JsonValue

from aicore.evals.models import CaseResult, EvalCase, EvalReport
from aicore.evals.scorers import Scorer
from aicore.tracing import InMemoryTracer

_log = logging.getLogger(__name__)

Target = Callable[[EvalCase], Awaitable[JsonValue]]
"""The thing under test: takes one case, returns its output."""


async def run_eval(
    name: str,
    cases: Sequence[EvalCase],
    target: Target,
    scorers: Sequence[Scorer],
    *,
    concurrency: int = 4,
    tracer: InMemoryTracer | None = None,
) -> EvalReport:
    """Run ``target`` on every case, score each output, return a report.

    At most ``concurrency`` cases run at the same time, which keeps a
    run under the provider's rate limit. Results are in dataset order,
    whatever order the cases finish in.

    If the target raises, the case gets ``error`` set to the exception
    class name and no scores: a timeout or a blocked request is a
    failed case, not a failed run. A scorer that raises is a bug in the
    eval code and is not caught.

    Raises:
        ValueError: If ``concurrency`` is less than 1.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, got {concurrency}")

    started_at = datetime.now(UTC)
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(case: EvalCase) -> CaseResult:
        """Run and score one case while holding a semaphore slot."""
        async with semaphore:
            return await _run_case(case, target, scorers)

    first_record = len(tracer.records) if tracer is not None else 0
    results = await asyncio.gather(*(run_one(case) for case in cases))
    calls = tracer.records[first_record:] if tracer is not None else []
    costs = [call.cost_usd for call in calls if call.cost_usd is not None]
    return EvalReport(
        name=name,
        started_at=started_at,
        results=list(results),
        llm_calls=len(calls),
        total_cost_usd=sum(costs) if costs else None,
    )


async def _run_case(case: EvalCase, target: Target, scorers: Sequence[Scorer]) -> CaseResult:
    """Call the target once for ``case`` and score its output.

    Catches ``Exception`` only, so ``CancelledError`` and
    ``KeyboardInterrupt`` (both ``BaseException``) still stop the run.
    Latency covers the target call only, not scoring.
    """
    start = time.perf_counter()
    try:
        output = await target(case)
    except Exception as err:
        latency_ms = (time.perf_counter() - start) * 1000
        # Class name only: the message may contain the prompt or user data.
        _log.warning("eval case %s failed: %s", case.id, type(err).__name__)
        return CaseResult(
            case_id=case.id,
            error=type(err).__name__,
            latency_ms=latency_ms,
            tags=case.tags,
        )
    latency_ms = (time.perf_counter() - start) * 1000
    return CaseResult(
        case_id=case.id,
        output=output,
        scores=[scorer.score(case, output) for scorer in scorers],
        latency_ms=latency_ms,
        tags=case.tags,
    )
