"""The golden dataset: conversations of questions, each with the SQL that answers it.

One conversation per line of ``evals/golden.jsonl``. A single question is
a conversation with one turn. In longer conversations, later turns only
make sense with the earlier ones ("Only for 2024."), and their gold SQL
is the full query the follow-up means.

Gold queries are written so their answer is unambiguous: no ties at a
``LIMIT`` cut-off and no ties inside an ORDER BY that the eval compares
in order. ``check_gold`` runs every gold query, so a failed case is
always the system's fault, not the dataset's.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from aicore.evals.dataset import DatasetError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from text_to_sql_agent.executor import QueryResult, QueryRunner, SqlExecutionError
from text_to_sql_agent.guard import UnsafeSqlError, check_sql

Split = Literal["dev", "test"]


class GoldTurn(BaseModel):
    """One question and the SQL whose result is the right answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str = Field(min_length=1)
    sql: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


class GoldenCase(BaseModel):
    """A conversation; one turn for a stand-alone question."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    split: Split
    turns: list[GoldTurn] = Field(min_length=1)


def load_golden(path: Path, split: Split | None = None) -> list[GoldenCase]:
    """Load the cases in ``path``, optionally only one ``split``.

    Raises:
        DatasetError: If a line is not a valid case or an id repeats,
            naming the file and line.
    """
    cases: list[GoldenCase] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = GoldenCase.model_validate_json(line)
        except ValidationError as err:
            raise DatasetError(f"{path}:{lineno}: {err.error_count()} error(s): {err}") from err
        if case.id in seen:
            raise DatasetError(f"{path}:{lineno}: duplicate id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    return [c for c in cases if split is None or c.split == split]


def guard_problems(cases: Sequence[GoldenCase]) -> list[str]:
    """Gold queries the guard would reject; checked without a database."""
    problems = []
    for case in cases:
        for i, turn in enumerate(case.turns, start=1):
            try:
                check_sql(turn.sql)
            except UnsafeSqlError as err:
                problems.append(f"{case.id} turn {i}: {err}")
    return problems


async def run_gold(
    cases: Sequence[GoldenCase], executor: QueryRunner
) -> tuple[dict[tuple[str, int], QueryResult], list[str]]:
    """Run every gold query once.

    Returns:
        The results keyed by ``(case id, turn index)``, and the problems
        found: a query that fails, returns no rows, or is cut off at the
        executor's row limit. An empty-result gold query would let a
        wrong query that also returns nothing pass.
    """
    results: dict[tuple[str, int], QueryResult] = {}
    problems: list[str] = []
    for case in cases:
        for i, turn in enumerate(case.turns):
            where = f"{case.id} turn {i + 1}"
            try:
                result = await executor.run(turn.sql)
            except (UnsafeSqlError, SqlExecutionError) as err:
                problems.append(f"{where}: {err}")
                continue
            if not result.rows:
                problems.append(f"{where}: returns no rows")
            elif result.truncated:
                problems.append(f"{where}: more rows than the executor returns")
            results[(case.id, i)] = result
    return results, problems
