"""Tests for the conversation eval: per-turn scoring, threads, errors and the report."""

from datetime import UTC, datetime
from pathlib import Path

from aicore.errors import ProviderTimeoutError
from aicore.tracing import CallRecord, InMemoryTracer
from text_to_sql_agent.evaluate import EvalRun, evaluate, format_run
from text_to_sql_agent.executor import QueryResult
from text_to_sql_agent.golden import GoldenCase
from text_to_sql_agent.turn import AgentTurn


def _case(cid: str, *sqls: str) -> GoldenCase:
    turns = [
        {"question": f"q{i}", "sql": sql, "tags": ["follow-up"] if i else ["join"]}
        for i, sql in enumerate(sqls)
    ]
    return GoldenCase.model_validate({"id": cid, "split": "dev", "turns": turns})


def _rows(*values: int) -> QueryResult:
    return QueryResult(columns=["x"], rows=[[v] for v in values])


def _record() -> CallRecord:
    now = datetime.now(UTC)
    return CallRecord(
        operation="complete",
        provider="fake",
        model="m",
        started_at=now,
        latency_ms=1.0,
        attempts=1,
        outcome="ok",
        cost_usd=0.001,
    )


class _ScriptedAgent:
    """Answers each question from a script and logs one LLM call per question."""

    name = "scripted"

    def __init__(self, tracer: InMemoryTracer, answers: dict[str, AgentTurn | Exception]):
        self.tracer = tracer
        self.answers = answers
        self.threads: list[str] = []

    async def ask(self, thread_id: str, question: str) -> AgentTurn:
        self.threads.append(thread_id)
        self.tracer.record(_record())
        answer = self.answers[question]
        if isinstance(answer, Exception):
            raise answer
        return answer


async def test_turns_are_scored_against_their_own_gold_result_in_one_thread() -> None:
    tracer = InMemoryTracer()
    case = _case("c1", "SELECT 1", "SELECT 2 ORDER BY 1")
    gold = {("c1", 0): _rows(1, 2), ("c1", 1): _rows(3, 4)}
    agent = _ScriptedAgent(
        tracer,
        {
            "q0": AgentTurn(question="q0", sql="s0", result=_rows(2, 1)),  # order-free: ok
            "q1": AgentTurn(question="q1", sql="s1", result=_rows(4, 3)),  # gold sorts: wrong
        },
    )
    [result] = await evaluate([case], agent, gold, tracer)
    first, second = result.turns
    assert first.correct and first.rows == 2 and first.llm_calls == 1
    assert first.cost_usd == 0.001
    assert not second.correct and second.sql == "s1"
    assert not result.correct
    assert len(set(agent.threads)) == 1 and agent.threads[0].startswith("eval-c1-")


async def test_each_conversation_gets_a_fresh_thread() -> None:
    tracer = InMemoryTracer()
    agent = _ScriptedAgent(tracer, {"q0": AgentTurn(question="q0", sql="s", result=_rows(1))})
    gold = {("a", 0): _rows(1), ("b", 0): _rows(1)}
    await evaluate([_case("a", "SELECT 1"), _case("b", "SELECT 1")], agent, gold, tracer)
    assert agent.threads[0] != agent.threads[1]


async def test_agent_errors_and_model_failures_are_recorded_and_the_conversation_goes_on() -> None:
    tracer = InMemoryTracer()
    case = _case("c", "SELECT 1", "SELECT 2", "SELECT 3")
    gold = {("c", i): _rows(1) for i in range(3)}
    agent = _ScriptedAgent(
        tracer,
        {
            "q0": ProviderTimeoutError("slow"),
            "q1": AgentTurn(question="q1", sql="DELETE", error_kind="unsafe", error="no"),
            "q2": AgentTurn(question="q2", sql="SELECT 1", result=_rows(1)),
        },
    )
    [result] = await evaluate([case], agent, gold, tracer)
    assert [t.error_kind for t in result.turns] == ["llm", "unsafe", None]
    assert result.turns[0].error == "ProviderTimeoutError"
    assert [t.correct for t in result.turns] == [False, False, True]


async def test_report_splits_first_turns_from_follow_ups_and_saves_json(tmp_path: Path) -> None:
    tracer = InMemoryTracer()
    single = _case("s", "SELECT 1")
    convo = _case("c", "SELECT 1", "SELECT 2")
    gold = {("s", 0): _rows(1), ("c", 0): _rows(1), ("c", 1): _rows(2)}
    agent = _ScriptedAgent(
        tracer,
        {
            "q0": AgentTurn(question="q0", sql="SELECT 1", result=_rows(1)),
            "q1": AgentTurn(question="q1", sql="SELECT 9", result=_rows(9)),
        },
    )
    cases = await evaluate([single, convo], agent, gold, tracer)
    run = EvalRun(
        agent="scripted",
        model="m",
        sql_prompt_version="v1",
        split="dev",
        created_at=datetime.now(UTC),
        cases=cases,
    )
    text = format_run(run)
    assert "| Execution accuracy, all turns | 0.67 (2/3) |" in text
    assert "| First turns (stand-alone questions) | 1.00 (2/2) |" in text
    assert "| Follow-up turns | 0.00 (0/1) |" in text
    assert "| Conversations fully correct (2+ turns) | 0.00 (0/1) |" in text
    assert "follow-up 0.00 (0/1)" in text
    assert "- BAD c.2 q1" in text and "sql: SELECT 9" in text

    path = run.save(tmp_path)
    assert path.name == "scripted-dev.json"
    assert EvalRun.model_validate_json(path.read_text(encoding="utf-8")) == run
