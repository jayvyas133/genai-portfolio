"""Tests for the single-shot baseline agent."""

from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.providers.fakeai import FakeProvider
from text_to_sql_agent.baseline import SingleShotAgent
from text_to_sql_agent.executor import QueryResult, SqlExecutionError
from text_to_sql_agent.guard import check_sql
from text_to_sql_agent.schema import Column, Schema, Table

SCHEMA = Schema(
    tables=[
        Table(
            name="genre",
            columns=[Column(name="name", type="text", not_null=False)],
            constraints=[],
        )
    ]
)


class _Runner:
    """Fake ``QueryRunner``: guards like the real one, then returns or raises."""

    def __init__(self, result: QueryResult | Exception) -> None:
        self.result = result
        self.queries: list[str] = []

    async def run(self, text: str) -> QueryResult:
        self.queries.append(check_sql(text))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _agent(reply: str, runner: _Runner) -> tuple[SingleShotAgent, FakeProvider]:
    fake = FakeProvider([reply])
    return SingleShotAgent(LLMClient(fake, LLMSettings(_env_file=None)), runner, SCHEMA), fake


async def test_writes_sql_against_the_schema_and_returns_the_rows() -> None:
    rows = QueryResult(columns=["n"], rows=[[25]])
    runner = _Runner(rows)
    agent, fake = _agent("```sql\nSELECT count(*) AS n FROM genre;\n```", runner)
    turn = await agent.ask("t1", "How many genres?")
    assert turn.sql == "SELECT count(*) AS n FROM genre;"
    assert turn.result == rows and turn.error_kind is None
    assert runner.queries == ["SELECT count(*) AS n FROM genre"]
    assert "CREATE TABLE genre" in fake.calls[0].messages[0].content


async def test_a_reply_without_sql_runs_nothing() -> None:
    runner = _Runner(QueryResult(columns=[], rows=[]))
    agent, _ = _agent("Sorry, I can't.", runner)
    turn = await agent.ask("t1", "q")
    assert turn.error_kind == "no_sql" and turn.sql is None
    assert runner.queries == []


async def test_guard_rejections_and_database_errors_are_reported_not_raised() -> None:
    agent, _ = _agent("```sql\nDELETE FROM genre\n```", _Runner(QueryResult(columns=[], rows=[])))
    unsafe = await agent.ask("t1", "q")
    assert unsafe.error_kind == "unsafe" and "DELETE" in (unsafe.error or "")

    failing = _Runner(SqlExecutionError('column "x" does not exist (SQLSTATE 42703)', "42703"))
    agent, _ = _agent("```sql\nSELECT x FROM genre\n```", failing)
    turn = await agent.ask("t1", "q")
    assert turn.error_kind == "execution" and "42703" in (turn.error or "")
    assert turn.sql == "SELECT x FROM genre"


async def test_has_no_memory_each_question_is_sent_alone() -> None:
    fake = FakeProvider(["```sql\nSELECT 1\n```", "```sql\nSELECT 2\n```"])
    runner = _Runner(QueryResult(columns=["x"], rows=[[1]]))
    agent = SingleShotAgent(LLMClient(fake, LLMSettings(_env_file=None)), runner, SCHEMA)
    await agent.ask("t1", "first question")
    await agent.ask("t1", "Only for 2024.")
    assert [len(c.messages) for c in fake.calls] == [2, 2]
    assert fake.calls[1].messages[-1].content == "Only for 2024."
