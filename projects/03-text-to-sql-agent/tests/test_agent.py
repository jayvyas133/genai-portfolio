"""Tests for the conversational agent graph: routing, repair, memory and budgets."""

from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.providers.fakeai import FakeProvider
from text_to_sql_agent.agent import (
    AgentLimits,
    ConversationalAgent,
    PastTurn,
    build_sql_messages,
    format_rows,
)
from text_to_sql_agent.executor import QueryResult, SqlExecutionError
from text_to_sql_agent.guard import check_sql
from text_to_sql_agent.schema import Column, Schema, Table

SCHEMA = Schema(
    tables=[
        Table(
            name="invoice",
            columns=[Column(name="total", type="numeric", not_null=True)],
            constraints=[],
        )
    ]
)
ROWS = QueryResult(columns=["country", "revenue"], rows=[["USA", 523.06], ["Canada", 303.96]])


class _Runner:
    """Fake ``QueryRunner``: guards like the real one, then answers per SQL text."""

    def __init__(self, answers: dict[str, QueryResult | Exception] | None = None) -> None:
        self.answers = answers or {}
        self.queries: list[str] = []

    async def run(self, text: str) -> QueryResult:
        sql = check_sql(text)
        self.queries.append(sql)
        answer = self.answers.get(sql, ROWS)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _sql(text: str) -> str:
    return f"```sql\n{text}\n```"


def _agent(
    script: list[str], runner: _Runner | None = None, limits: AgentLimits | None = None
) -> tuple[ConversationalAgent, FakeProvider, _Runner]:
    fake = FakeProvider(script)
    runner = runner or _Runner()
    client = LLMClient(fake, LLMSettings(_env_file=None))
    return ConversationalAgent(client, runner, SCHEMA, limits), fake, runner


async def test_success_runs_the_sql_and_answers_in_two_calls() -> None:
    agent, fake, runner = _agent([_sql("SELECT 1"), "USA leads with 523.06."])
    turn = await agent.ask("t", "Revenue by country?")
    assert turn.sql == "SELECT 1" and turn.result == ROWS
    assert turn.answer == "USA leads with 523.06." and turn.attempts == 1
    assert turn.error_kind is None
    assert runner.queries == ["SELECT 1"] and len(fake.calls) == 2
    answer_prompt = fake.calls[1].messages[-1].content
    assert "Revenue by country?" in answer_prompt and "USA | 523.06" in answer_prompt


async def test_a_database_error_goes_back_to_the_model_and_the_repair_runs() -> None:
    error = SqlExecutionError('column "t.title" does not exist (SQLSTATE 42703)', "42703")
    runner = _Runner({"SELECT t.title FROM t": error})
    agent, fake, _ = _agent(
        [_sql("SELECT t.title FROM t"), _sql("SELECT a.title FROM a"), "Done."], runner
    )
    turn = await agent.ask("t", "Album titles?")
    assert turn.sql == "SELECT a.title FROM a" and turn.attempts == 2
    assert turn.result == ROWS and turn.error is None
    repair = fake.calls[1].messages
    assert repair[-2].role == "assistant" and "SELECT t.title FROM t" in repair[-2].content
    assert repair[-1].role == "user" and "42703" in repair[-1].content


async def test_guard_rejections_and_replies_without_sql_are_repaired_too() -> None:
    agent, fake, runner = _agent(
        ["I think you want revenue.", _sql("DELETE FROM x"), _sql("SELECT 2"), "ok"]
    )
    turn = await agent.ask("t", "q")
    assert turn.attempts == 3 and turn.sql == "SELECT 2" and turn.result is not None
    assert runner.queries == ["SELECT 2"]  # the DELETE never reached the runner
    third = fake.calls[2].messages
    assert "no SQL query" in third[-3].content
    assert "Only SELECT queries are allowed" in third[-1].content


async def test_out_of_attempts_apologises_without_an_answer_call() -> None:
    error = SqlExecutionError("syntax error (SQLSTATE 42601)", "42601")
    runner = _Runner({"SELECT bad": error})
    agent, fake, _ = _agent([_sql("SELECT bad")] * 5, runner, AgentLimits(max_attempts=2))
    turn = await agent.ask("t", "q")
    assert turn.attempts == 2 and turn.result is None
    assert turn.error_kind == "execution" and "42601" in (turn.error or "")
    assert turn.answer is not None and turn.answer.startswith("Sorry")
    assert len(fake.calls) == 2  # two SQL attempts, no answer call


async def test_an_empty_result_is_answered_without_a_call() -> None:
    runner = _Runner({"SELECT 1": QueryResult(columns=["x"], rows=[])})
    agent, fake, _ = _agent([_sql("SELECT 1")], runner)
    turn = await agent.ask("t", "q")
    assert turn.answer == "No rows matched that question." and len(fake.calls) == 1


async def test_follow_ups_see_earlier_turns_of_the_same_thread_only() -> None:
    agent, fake, _ = _agent(
        [
            _sql("SELECT revenue"),
            "a1",
            _sql("SELECT revenue WHERE 2024"),
            "a2",
            _sql("SELECT x"),
            "a3",
        ]
    )
    first = await agent.ask("t1", "Revenue by country?")
    second = await agent.ask("t1", "Only for 2024.")
    other = await agent.ask("t2", "Only for 2024.")
    assert first.attempts == second.attempts == other.attempts == 1

    follow_up = [m.content for m in fake.calls[2].messages]
    assert follow_up[1:] == [
        "Revenue by country?",
        f"{_sql('SELECT revenue')}\nResult: {format_rows(ROWS, 5)}",
        "Only for 2024.",
    ]
    fresh = [m.role for m in fake.calls[4].messages]
    assert fresh == ["system", "user"]  # thread t2 starts with no history


async def test_per_turn_state_is_reset_between_turns() -> None:
    error = SqlExecutionError("boom (SQLSTATE 42601)", "42601")
    runner = _Runner({"SELECT bad": error})
    agent, fake, _ = _agent(
        [_sql("SELECT bad"), _sql("SELECT good"), "a1", _sql("SELECT next"), "a2"], runner
    )
    await agent.ask("t", "q1")
    turn = await agent.ask("t", "q2")
    assert turn.attempts == 1 and turn.error is None
    # The second turn's prompt holds no repair messages from the first turn.
    assert "failed" not in fake.calls[3].messages[-1].content


async def test_history_is_trimmed_to_the_window() -> None:
    script = [s for i in range(4) for s in (_sql(f"SELECT {i}"), f"a{i}")]
    agent, fake, _ = _agent(script + [_sql("SELECT 9"), "a9"], limits=AgentLimits(history_turns=2))
    for i in range(4):
        await agent.ask("t", f"q{i}")
    await agent.ask("t", "last")
    users = [m.content for m in fake.calls[-2].messages if m.role == "user"]
    assert users == ["q2", "q3", "last"]


async def test_failed_turns_are_remembered_with_their_error() -> None:
    error = SqlExecutionError("boom (SQLSTATE 42601)", "42601")
    agent, fake, _ = _agent(
        [_sql("SELECT bad"), _sql("SELECT ok"), "a"],
        _Runner({"SELECT bad": error}),
        AgentLimits(max_attempts=1),
    )
    await agent.ask("t", "q1")
    await agent.ask("t", "q2")
    assert "Result: failed: boom" in fake.calls[1].messages[2].content


async def test_a_zero_window_sends_no_history() -> None:
    agent, fake, _ = _agent(
        [_sql("SELECT 1"), "a", _sql("SELECT 2"), "b"], limits=AgentLimits(history_turns=0)
    )
    await agent.ask("t", "q1")
    await agent.ask("t", "q2")
    assert [m.role for m in fake.calls[2].messages] == ["system", "user"]


def test_rows_preview_shows_how_many_rows_there_were() -> None:
    many = QueryResult(columns=["x"], rows=[[i] for i in range(8)], truncated=True)
    assert format_rows(many, 2) == "(2 of 8+ rows)\nx\n0\n1"


def test_a_past_turn_without_sql_is_shown_as_no_query() -> None:
    past = PastTurn(question="q", sql=None, preview="failed: x")
    request = build_sql_messages("DDL", [past], "next", [])
    assert request.messages[2].content == "(no query)\nResult: failed: x"
