"""Score an agent on the golden conversations by execution accuracy.

The eval drives the agent the way a chat client would: a fresh
conversation id per golden case, then each question in order. Every
turn is scored with the agent's *own* earlier answers as its history,
not the gold ones, as in the CoSQL and SParC multi-turn benchmarks. That
is what a user experiences: an early mistake can derail a follow-up.

A turn is correct when its result matches the gold query's result (see
``compare.py``); no LLM judge is involved, so scores are cheap,
deterministic and comparable across runs. A conversation is correct
only when every turn is.

``aicore.evals.run_eval`` is not used: it scores independent cases,
while here turns depend on earlier turns of the same conversation.
"""

import math
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

from aicore.errors import LLMError
from aicore.tracing import InMemoryTracer
from pydantic import BaseModel, ConfigDict, Field

from text_to_sql_agent.compare import results_match
from text_to_sql_agent.executor import QueryResult
from text_to_sql_agent.golden import GoldenCase, Split
from text_to_sql_agent.guard import has_order_by
from text_to_sql_agent.turn import Agent, ErrorKind

TurnError = ErrorKind | Literal["llm"]


class TurnResult(BaseModel):
    """What happened on one turn, and whether its answer was right."""

    model_config = ConfigDict(frozen=True)

    question: str
    tags: list[str]
    gold_sql: str
    sql: str | None = None
    correct: bool = False
    rows: int | None = None
    error_kind: TurnError | None = None
    error: str | None = None
    latency_ms: float = Field(default=0.0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)


class CaseResult(BaseModel):
    """One golden conversation's turns, in order."""

    model_config = ConfigDict(frozen=True)

    id: str
    turns: list[TurnResult]

    @property
    def correct(self) -> bool:
        """Every turn was answered correctly."""
        return all(t.correct for t in self.turns)


class EvalRun(BaseModel):
    """One evaluation run: what produced it, and every case result."""

    model_config = ConfigDict(frozen=True)

    agent: str
    model: str
    sql_prompt_version: str
    split: Split | None
    created_at: datetime
    cases: list[CaseResult]

    def save(self, folder: Path) -> Path:
        """Write the run to ``folder/{agent}-{split}.json`` and return the path.

        A fixed name per agent and split means a re-run overwrites the
        file and ``git diff`` shows which turns changed.
        """
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self.agent}-{self.split or 'all'}.json"
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path


async def evaluate(
    cases: Sequence[GoldenCase],
    agent: Agent,
    gold: Mapping[tuple[str, int], QueryResult],
    tracer: InMemoryTracer,
) -> list[CaseResult]:
    """Run every conversation through ``agent`` and score each turn against ``gold``.

    Turns run one at a time so latencies are not distorted by
    concurrency. ``tracer`` must be the one the agent's client reports
    to; it gives each turn's call count and cost. A failed model call is
    recorded on its turn and the conversation continues.

    Args:
        cases: The golden conversations.
        agent: The system under test.
        gold: Gold results keyed by ``(case id, turn index)``, from
            ``golden.run_gold``.
        tracer: Shared with the agent's ``LLMClient``.
    """
    results: list[CaseResult] = []
    for case in cases:
        thread_id = f"eval-{case.id}-{uuid.uuid4().hex[:8]}"
        turns: list[TurnResult] = []
        for i, gold_turn in enumerate(case.turns):
            base = TurnResult(
                question=gold_turn.question, tags=gold_turn.tags, gold_sql=gold_turn.sql
            )
            before = len(tracer.records)
            start = time.perf_counter()
            try:
                answer = await agent.ask(thread_id, gold_turn.question)
            except LLMError as err:
                scored = base.model_copy(update={"error_kind": "llm", "error": type(err).__name__})
            else:
                correct = answer.result is not None and results_match(
                    answer.result, gold[(case.id, i)], ordered=has_order_by(gold_turn.sql)
                )
                scored = base.model_copy(
                    update={
                        "sql": answer.sql,
                        "correct": correct,
                        "rows": len(answer.result.rows) if answer.result else None,
                        "error_kind": answer.error_kind,
                        "error": answer.error,
                    }
                )
            records = tracer.records[before:]
            turns.append(
                scored.model_copy(
                    update={
                        "latency_ms": (time.perf_counter() - start) * 1000,
                        "llm_calls": len(records),
                        "cost_usd": sum(r.cost_usd or 0.0 for r in records),
                    }
                )
            )
        results.append(CaseResult(id=case.id, turns=turns))
    return results


def _ratio(num: int, den: int) -> str:
    """``0.83 (5/6)``, or ``-`` when there is nothing to divide by."""
    return f"{num / den:.2f} ({num}/{den})" if den else "-"


def _percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile (an observed value), or 0.0 for no values."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[max(math.ceil(pct / 100 * len(ordered)) - 1, 0)]


def _accuracy(turns: Sequence[TurnResult]) -> str:
    return _ratio(sum(t.correct for t in turns), len(turns))


def format_run(run: EvalRun) -> str:
    """Render ``run`` as Markdown: accuracy table, errors, latency and cost, then per turn."""
    turns = [t for c in run.cases for t in c.turns]
    first = [c.turns[0] for c in run.cases]
    follow_ups = [t for c in run.cases for t in c.turns[1:]]
    conversations = [c for c in run.cases if len(c.turns) > 1]
    rows = [
        ("Execution accuracy, all turns", _accuracy(turns)),
        ("First turns (stand-alone questions)", _accuracy(first)),
        ("Follow-up turns", _accuracy(follow_ups)),
        (
            "Conversations fully correct (2+ turns)",
            _ratio(sum(c.correct for c in conversations), len(conversations)),
        ),
        ("Turns whose SQL ran", _ratio(sum(t.rows is not None for t in turns), len(turns))),
    ]
    tags = sorted({tag for t in turns for tag in t.tags})
    errors = {
        kind: sum(t.error_kind == kind for t in turns)
        for kind in ("no_sql", "unsafe", "execution", "llm")
    }
    n = len(turns)
    cost = sum(t.cost_usd for t in turns)
    latencies = [t.latency_ms for t in turns]
    lines = [
        f"Agent: {run.agent} | model: {run.model} | prompt {run.sql_prompt_version} | "
        f"split: {run.split or 'all'} | {len(run.cases)} conversations, {n} turns",
        "",
        "| Metric | Value |",
        "|---|---|",
        *(f"| {label} | {value} |" for label, value in rows),
        "",
        "By tag: "
        + ", ".join(f"{tag} {_accuracy([t for t in turns if tag in t.tags])}" for tag in tags),
        "Errors: " + ", ".join(f"{k} {v}" for k, v in errors.items()),
        f"Latency per turn: p50 {_percentile(latencies, 50) / 1000:.1f} s, "
        f"p95 {_percentile(latencies, 95) / 1000:.1f} s",
        f"LLM calls per turn: {sum(t.llm_calls for t in turns) / n if n else 0:.1f}",
        f"Cost: ${cost / n if n else 0:.5f} per turn (${cost:.4f} total)",
        "",
        "Per turn:",
    ]
    for c in run.cases:
        for i, t in enumerate(c.turns, start=1):
            mark = "ok " if t.correct else "BAD"
            detail = f" [{t.error_kind}: {t.error}]" if t.error_kind else ""
            lines.append(f"- {mark} {c.id}.{i} {t.question}{detail}")
            if not t.correct and t.sql:
                lines.append(f"      sql: {' '.join(t.sql.split())}")
    return "\n".join(lines)
