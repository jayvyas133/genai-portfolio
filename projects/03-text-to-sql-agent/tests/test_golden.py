"""Tests for the golden dataset file and the gold-query runner."""

from pathlib import Path

import pytest
from aicore.evals.dataset import DatasetError
from text_to_sql_agent.config import TextToSqlSettings
from text_to_sql_agent.executor import QueryResult, SqlExecutionError
from text_to_sql_agent.golden import GoldenCase, guard_problems, load_golden, run_gold

GOLDEN = TextToSqlSettings(_env_file=None).evals_dir / "golden.jsonl"


def test_the_real_dataset_loads_with_both_splits_and_follow_ups() -> None:
    cases = load_golden(GOLDEN)
    turns = [t for c in cases for t in c.turns]
    assert len(cases) == 28 and len(turns) == 42
    assert {c.split for c in cases} == {"dev", "test"}
    assert all("follow-up" in t.tags for c in cases for t in c.turns[1:])
    assert not any("follow-up" in c.turns[0].tags for c in cases)
    assert len(load_golden(GOLDEN, "test")) < len(load_golden(GOLDEN, "dev"))


def test_every_real_gold_query_passes_the_guard() -> None:
    assert guard_problems(load_golden(GOLDEN)) == []


def test_bad_lines_and_duplicate_ids_name_the_line(tmp_path: Path) -> None:
    line = '{"id": "a", "split": "dev", "turns": [{"question": "q", "sql": "SELECT 1"}]}'
    path = tmp_path / "g.jsonl"
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(DatasetError, match=r"g.jsonl:2: duplicate id 'a'"):
        load_golden(path)
    path.write_text('{"id": "a", "split": "dev", "turns": []}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match=r"g.jsonl:1"):
        load_golden(path)


class _Runner:
    """Fake ``QueryRunner``: answers from a table of SQL to results or errors."""

    def __init__(self, answers: dict[str, QueryResult | Exception]) -> None:
        self.answers = answers

    async def run(self, text: str) -> QueryResult:
        answer = self.answers[text]
        if isinstance(answer, Exception):
            raise answer
        return answer


async def test_run_gold_keys_results_by_turn_and_reports_bad_queries() -> None:
    case = GoldenCase.model_validate(
        {
            "id": "c",
            "split": "dev",
            "turns": [
                {"question": "ok", "sql": "SELECT 1"},
                {"question": "empty", "sql": "SELECT 2"},
                {"question": "cut", "sql": "SELECT 3"},
                {"question": "broken", "sql": "SELECT 4"},
            ],
        }
    )
    one = QueryResult(columns=["x"], rows=[[1]])
    runner = _Runner(
        {
            "SELECT 1": one,
            "SELECT 2": QueryResult(columns=["x"], rows=[]),
            "SELECT 3": QueryResult(columns=["x"], rows=[[1]], truncated=True),
            "SELECT 4": SqlExecutionError("column does not exist"),
        }
    )
    results, problems = await run_gold([case], runner)
    assert results[("c", 0)] == one
    assert ("c", 3) not in results
    assert problems == [
        "c turn 2: returns no rows",
        "c turn 3: more rows than the executor returns",
        "c turn 4: column does not exist",
    ]
