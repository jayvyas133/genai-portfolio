"""The conversational Text-to-SQL agent: a small LangGraph graph run once per chat turn.

    START -> write_sql -> execute -> answer -> record -> END
                ^            |
                +-- error ---+  (until max_attempts)
                             +-- out of attempts -> record

- **Memory.** A LangGraph checkpointer keeps each conversation's state
  under its ``thread_id``. The last ``history_turns`` turns (question, SQL
  and a few result rows) go into the prompt as earlier chat messages, so
  "Only for 2024." can be answered by changing the previous query.
- **Self-correction.** When the guard or Postgres rejects the SQL, the
  failed query and the exact error go back to the model as the next chat
  message. Database errors name the problem (``column t.title does not
  exist``), so the model can usually fix it in one try.
- **Answer.** One more call turns the rows into a short reply in plain
  English. Rows and SQL are returned too, and the eval scores the rows.

Budget: at most ``max_attempts`` SQL-writing calls plus one answer call
per turn. A turn that fails every attempt gets an apology written in
code, with no extra call.

State holds only plain data (str, int, list, dict), so any checkpointer
can store it, including the Postgres one the M3 API will use.
"""

from dataclasses import dataclass
from typing import Any, TypedDict

from aicore.client import LLMClient
from aicore.models import LLMRequest, Message
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from text_to_sql_agent.executor import QueryResult, QueryRunner, SqlExecutionError
from text_to_sql_agent.generation import extract_sql
from text_to_sql_agent.guard import UnsafeSqlError
from text_to_sql_agent.schema import Schema
from text_to_sql_agent.turn import AgentTurn, ErrorKind

AGENT_PROMPT_VERSION = "v1"
"""Saved with eval results; bump it whenever a prompt here changes."""

SQL_SYSTEM_PROMPT = """\
You are a data assistant. You answer questions about a PostgreSQL \
database by writing SQL.

Write one read-only PostgreSQL SELECT query that answers the latest \
question. Use only the tables and columns in the schema below. Select \
only the columns the question asks for, with clear names.

If the latest question follows up on an earlier one, start from the \
earlier query and change only what the new question asks for. If it \
stands on its own, ignore the earlier queries.

Reply with the query in a ```sql code block and nothing else.

Database schema:

{schema}"""

ANSWER_SYSTEM_PROMPT = """\
You answer a user's question from the result of a SQL query. Use only \
the rows given; do not guess beyond them. Reply in one to three short \
sentences and include the key names and numbers."""

REPAIR_MESSAGE = "That query failed: {error}\nWrite a corrected query."


@dataclass(frozen=True)
class AgentLimits:
    """Budgets per turn and how much of the conversation goes into prompts."""

    max_attempts: int = 3
    history_turns: int = 3
    preview_rows: int = 5
    answer_rows: int = 20


class PastTurn(TypedDict):
    """One finished turn as remembered: what was asked, the SQL, and a glimpse of the result."""

    question: str
    sql: str | None
    preview: str  # a few result rows, or the error


class Attempt(TypedDict):
    """One SQL-writing attempt in the current turn."""

    sql: str | None
    error: str


class TurnState(TypedDict, total=False):
    """The graph state. ``history`` persists across turns; the rest is reset each turn."""

    history: list[PastTurn]
    question: str
    sql: str | None
    attempts: int
    failed: list[Attempt]
    error_kind: ErrorKind | None
    error: str | None
    result: dict[str, Any] | None  # QueryResult.model_dump(), kept as plain data
    answer: str | None


def format_rows(result: QueryResult, limit: int) -> str:
    """The first ``limit`` rows as a compact pipe-separated table, with the row count."""
    lines = [" | ".join(result.columns)]
    lines += [
        " | ".join("NULL" if v is None else str(v) for v in row) for row in result.rows[:limit]
    ]
    total = f"{len(result.rows)}{'+' if result.truncated else ''}"
    shown = min(limit, len(result.rows))
    return f"({shown} of {total} rows)\n" + "\n".join(lines)


def build_sql_messages(
    ddl: str, history: list[PastTurn], question: str, failed: list[Attempt]
) -> LLMRequest:
    """The SQL-writing request: schema, earlier turns as chat, the question, then any failures.

    Earlier turns are replayed as user/assistant messages, the format chat
    models are trained on. Each failed attempt this turn is added as the
    model's reply followed by the error, so a repair sees everything it
    tried.
    """
    messages = [Message(role="system", content=SQL_SYSTEM_PROMPT.format(schema=ddl))]
    for past in history:
        messages.append(Message(role="user", content=past["question"]))
        reply = f"```sql\n{past['sql']}\n```" if past["sql"] else "(no query)"
        messages.append(Message(role="assistant", content=f"{reply}\nResult: {past['preview']}"))
    messages.append(Message(role="user", content=question))
    for attempt in failed:
        reply = f"```sql\n{attempt['sql']}\n```" if attempt["sql"] else "(no query)"
        messages.append(Message(role="assistant", content=reply))
        messages.append(Message(role="user", content=REPAIR_MESSAGE.format(error=attempt["error"])))
    return LLMRequest(messages=messages, temperature=0.0, max_tokens=512)


def build_answer_request(question: str, sql: str, rows: str) -> LLMRequest:
    """The answer request: the question, the SQL that ran, and its rows."""
    user = f"Question: {question}\n\nSQL:\n{sql}\n\nResult {rows}"
    return LLMRequest(
        messages=[
            Message(role="system", content=ANSWER_SYSTEM_PROMPT),
            Message(role="user", content=user),
        ],
        temperature=0.0,
        max_tokens=200,
    )


class ConversationalAgent:
    """``Agent`` with memory per ``thread_id``, self-correction and a plain-English answer."""

    name = "agent"

    def __init__(
        self,
        client: LLMClient,
        runner: QueryRunner,
        schema: Schema,
        limits: AgentLimits | None = None,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
    ) -> None:
        """Build and compile the graph once; it is reused for every turn.

        Args:
            client: Makes every model call.
            runner: Guards and runs SQL (``SqlExecutor`` in production).
            schema: The tables the model may query.
            limits: Budgets; defaults to ``AgentLimits()``.
            checkpointer: Where conversations are kept. Defaults to an
                in-memory store, which lives as long as this object.
        """
        self._client = client
        self._runner = runner
        self._ddl = schema.ddl()
        self._limits = limits or AgentLimits()
        self.graph = self._build(checkpointer or InMemorySaver())

    async def ask(self, thread_id: str, question: str) -> AgentTurn:
        """Answer ``question`` as the next turn of conversation ``thread_id``.

        Raises:
            LLMError: If a model call fails after retries.
        """
        # Per-turn fields are reset here; ``history`` comes from the checkpoint.
        start: TurnState = {
            "question": question,
            "sql": None,
            "attempts": 0,
            "failed": [],
            "error_kind": None,
            "error": None,
            "result": None,
            "answer": None,
        }
        state: TurnState = await self.graph.ainvoke(
            start, {"configurable": {"thread_id": thread_id}}
        )  # type: ignore[assignment]
        result = state.get("result")
        return AgentTurn(
            question=question,
            sql=state.get("sql"),
            result=QueryResult.model_validate(result) if result is not None else None,
            error_kind=state.get("error_kind"),
            error=state.get("error"),
            answer=state.get("answer"),
            attempts=state.get("attempts", 0),
        )

    async def history(self, thread_id: str) -> list[PastTurn]:
        """The turns remembered for ``thread_id`` (the last ``history_turns``), oldest first."""
        snapshot = await self.graph.aget_state({"configurable": {"thread_id": thread_id}})
        history: list[PastTurn] = snapshot.values.get("history", [])
        return history

    def _build(self, checkpointer: BaseCheckpointSaver[Any]) -> CompiledStateGraph:  # type: ignore[type-arg]
        graph = StateGraph(TurnState)
        graph.add_node("write_sql", self._write_sql)
        graph.add_node("execute", self._execute)
        graph.add_node("answer", self._answer)
        graph.add_node("record", self._record)
        graph.add_edge(START, "write_sql")
        graph.add_edge("write_sql", "execute")
        graph.add_conditional_edges(
            "execute", self._after_execute, ["answer", "write_sql", "record"]
        )
        graph.add_edge("answer", "record")
        graph.add_edge("record", END)
        return graph.compile(checkpointer=checkpointer)

    async def _write_sql(self, state: TurnState) -> TurnState:
        """Ask for SQL, with the conversation so far and this turn's failures."""
        request = build_sql_messages(
            self._ddl, state.get("history", []), state["question"], state.get("failed", [])
        )
        response = await self._client.complete(request)
        sql = extract_sql(response.text)
        return {"sql": sql, "attempts": state.get("attempts", 0) + 1}

    async def _execute(self, state: TurnState) -> TurnState:
        """Run the SQL through the guard and the read-only executor."""
        sql = state.get("sql")
        if sql is None:
            return self._failure(state, "no_sql", "The reply contained no SQL query.")
        try:
            result = await self._runner.run(sql)
        except UnsafeSqlError as err:
            return self._failure(state, "unsafe", str(err))
        except SqlExecutionError as err:
            return self._failure(state, "execution", str(err))
        return {"result": result.model_dump(), "error_kind": None, "error": None}

    def _failure(self, state: TurnState, kind: ErrorKind, error: str) -> TurnState:
        """Record a failed attempt so the next ``write_sql`` call sees it."""
        failed = [*state.get("failed", []), Attempt(sql=state.get("sql"), error=error)]
        return {"error_kind": kind, "error": error, "failed": failed, "result": None}

    def _after_execute(self, state: TurnState) -> str:
        """Answer on success, retry while attempts remain, otherwise give up."""
        if state.get("result") is not None:
            return "answer"
        if state.get("attempts", 0) < self._limits.max_attempts:
            return "write_sql"
        return "record"

    async def _answer(self, state: TurnState) -> TurnState:
        """Turn the rows into a short reply; no call is needed for an empty result."""
        result = QueryResult.model_validate(state["result"])
        if not result.rows:
            return {"answer": "No rows matched that question."}
        request = build_answer_request(
            state["question"], state.get("sql") or "", format_rows(result, self._limits.answer_rows)
        )
        response = await self._client.complete(request)
        return {"answer": response.text.strip()}

    async def _record(self, state: TurnState) -> TurnState:
        """Add this turn to the remembered history (trimmed) and apologise if it failed."""
        result = state.get("result")
        if result is not None:
            preview = format_rows(QueryResult.model_validate(result), self._limits.preview_rows)
            answer = state.get("answer")
        else:
            preview = f"failed: {state.get('error')}"
            answer = f"Sorry, I could not answer that. The last error was: {state.get('error')}"
        past = PastTurn(question=state["question"], sql=state.get("sql"), preview=preview)
        history = [*state.get("history", []), past]
        # Slice from the front: ``[-0:]`` would keep everything.
        history = history[max(len(history) - self._limits.history_turns, 0) :]
        return {"history": history, "answer": answer}
