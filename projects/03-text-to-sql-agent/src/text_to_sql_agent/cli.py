"""Command line for the Text-to-SQL agent.

Run with ``uv run text-to-sql <command>``. Settings come from environment
variables or ``.env`` (``TEXT_TO_SQL_*`` for this project, ``AICORE_*``
for the models).
"""

import argparse
import asyncio
import logging
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

import psycopg
from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.errors import LLMError
from aicore.providers.factory import build_client
from aicore.tracing import InMemoryTracer

from text_to_sql_agent import aio
from text_to_sql_agent.agent import AGENT_PROMPT_VERSION, ConversationalAgent
from text_to_sql_agent.baseline import SingleShotAgent
from text_to_sql_agent.config import TextToSqlSettings
from text_to_sql_agent.database import LoadError, load_chinook
from text_to_sql_agent.evaluate import EvalRun, evaluate, format_run
from text_to_sql_agent.executor import QueryResult, QueryRunner, SqlExecutor
from text_to_sql_agent.generation import SQL_PROMPT_VERSION
from text_to_sql_agent.golden import Split, load_golden, run_gold
from text_to_sql_agent.schema import Schema, load_schema
from text_to_sql_agent.turn import AgentTurn

AGENTS = ["agent", "single-shot"]


def _executor(settings: TextToSqlSettings) -> SqlExecutor:
    return SqlExecutor(settings.database_url, settings.statement_timeout_ms, settings.max_rows)


def build_agent(
    name: str,
    client: LLMClient,
    runner: QueryRunner,
    schema: Schema,
    settings: TextToSqlSettings,
) -> ConversationalAgent | SingleShotAgent:
    """The agent called ``name``, wired to ``client``, ``runner`` and ``schema``."""
    if name == "agent":
        return ConversationalAgent(client, runner, schema, settings.agent_limits())
    return SingleShotAgent(client, runner, schema)


def format_table(result: QueryResult, limit: int = 20) -> str:
    """A plain-text table of the first ``limit`` rows."""
    shown = result.rows[:limit]
    cells = [result.columns] + [["NULL" if v is None else str(v) for v in row] for row in shown]
    widths = [max(len(row[i]) for row in cells) for i in range(len(result.columns))]
    lines = [" | ".join(v.ljust(w) for v, w in zip(row, widths, strict=True)) for row in cells]
    lines.insert(1, "-+-".join("-" * w for w in widths))
    more = len(result.rows) - len(shown)
    if more or result.truncated:
        lines.append(
            f"... {more} more row(s) shown of {len(result.rows)}"
            + (" (truncated by the row limit)" if result.truncated else "")
        )
    return "\n".join(lines)


def run_load_db(settings: TextToSqlSettings, reset: bool) -> None:
    """Create the Chinook database and the read-only role."""
    loaded = load_chinook(
        settings.admin_database_url,
        settings.database_url,
        settings.chinook_sql,
        settings.statement_timeout_ms,
        reset=reset,
    )
    print("Loaded Chinook." if loaded else "Chinook already loaded; role and grants refreshed.")


async def run_schema(settings: TextToSqlSettings) -> None:
    """Print the schema exactly as the model sees it."""
    print((await load_schema(settings.database_url)).ddl())


async def run_check_golden(settings: TextToSqlSettings) -> None:
    """Run every gold query and report problems."""
    cases = load_golden(settings.evals_dir / "golden.jsonl")
    _, problems = await run_gold(cases, _executor(settings))
    if problems:
        raise SystemExit("Golden dataset problems:\n" + "\n".join(problems))
    turns = sum(len(c.turns) for c in cases)
    print(f"All {turns} gold queries in {len(cases)} conversations run and return rows.")


def print_turn(turn: AgentTurn) -> None:
    """Show the answer, then the SQL and rows behind it."""
    if turn.answer:
        print(turn.answer)
        print()
    print(turn.sql or "(no SQL)")
    if turn.attempts > 1:
        print(f"(after {turn.attempts} attempts)")
    print()
    if turn.result is not None:
        print(format_table(turn.result))
    else:
        print(f"{turn.error_kind}: {turn.error}")


def _usage(tracer: InMemoryTracer, before: int, start: float) -> str:
    records = tracer.records[before:]
    cost = sum(r.cost_usd or 0.0 for r in records)
    return f"{time.perf_counter() - start:.1f} s, {len(records)} LLM call(s), ${cost:.5f}"


async def run_ask(settings: TextToSqlSettings, llm: LLMSettings, question: str, name: str) -> None:
    """Answer one question and print the answer, SQL and rows."""
    tracer = InMemoryTracer()
    client = build_client(llm, tracer=tracer)
    schema = await load_schema(settings.database_url)
    agent = build_agent(name, client, _executor(settings), schema, settings)
    start = time.perf_counter()
    try:
        turn = await agent.ask("cli", question)
    finally:
        await client.aclose()
    print_turn(turn)
    print(f"\n{_usage(tracer, 0, start)}")


async def run_chat(settings: TextToSqlSettings, llm: LLMSettings) -> None:
    """An interactive conversation with the agent; an empty line or Ctrl+D ends it."""
    tracer = InMemoryTracer()
    client = build_client(llm, tracer=tracer)
    schema = await load_schema(settings.database_url)
    agent = build_agent("agent", client, _executor(settings), schema, settings)
    thread_id = f"chat-{uuid.uuid4().hex[:8]}"
    print(f"Conversation {thread_id}. Ask about the Chinook music store; empty line to quit.")
    try:
        while True:
            try:
                # In a thread, so the event loop (and the client's connections) stays alive.
                question = (await asyncio.to_thread(input, "\n> ")).strip()
            except EOFError:
                break
            if not question:
                break
            before, start = len(tracer.records), time.perf_counter()
            print_turn(await agent.ask(thread_id, question))
            print(f"\n[{_usage(tracer, before, start)}]")
    finally:
        await client.aclose()


async def run_eval(
    settings: TextToSqlSettings,
    llm: LLMSettings,
    name: str,
    split: Split | None,
    limit: int | None,
) -> None:
    """Evaluate agent ``name`` on the golden ``split``, print the table and save JSON."""
    cases = load_golden(settings.evals_dir / "golden.jsonl", split)[:limit]
    executor = _executor(settings)
    gold, problems = await run_gold(cases, executor)
    if problems:
        raise SystemExit("Golden dataset problems:\n" + "\n".join(problems))
    tracer = InMemoryTracer()
    client = build_client(llm, tracer=tracer)
    schema = await load_schema(settings.database_url)
    agent = build_agent(name, client, executor, schema, settings)
    try:
        results = await evaluate(cases, agent, gold, tracer)
    finally:
        await client.aclose()
    run = EvalRun(
        agent=agent.name,
        model=llm.model,
        prompt_version=AGENT_PROMPT_VERSION if name == "agent" else SQL_PROMPT_VERSION,
        split=split,
        created_at=datetime.now(UTC),
        cases=results,
    )
    path = run.save(settings.evals_dir / "results")
    print(format_run(run))
    print(f"\nSaved {path}")


def build_parser() -> argparse.ArgumentParser:
    """The ``text-to-sql`` argument parser."""
    parser = argparse.ArgumentParser(prog="text-to-sql", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    load = sub.add_parser("load-db", help="create the Chinook database and the read-only role")
    load.add_argument("--reset", action="store_true", help="drop and reload the database")

    sub.add_parser("schema", help="print the schema the model sees")
    sub.add_parser("check-golden", help="run every gold query against the database")

    ask = sub.add_parser("ask", help="answer one question and print the SQL and rows")
    ask.add_argument("question")
    ask.add_argument("--agent", choices=AGENTS, default="agent")

    sub.add_parser("chat", help="have a conversation with the agent")

    ev = sub.add_parser("eval", help="score an agent on the golden conversations")
    ev.add_argument("--agent", choices=AGENTS, default="agent")
    ev.add_argument("--split", choices=["dev", "test"], default=None)
    ev.add_argument("--limit", type=int, default=None, help="only the first N conversations")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for ``uv run text-to-sql``."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    settings = TextToSqlSettings()
    try:
        match args.command:
            case "load-db":
                run_load_db(settings, args.reset)
            case "schema":
                aio.run(run_schema(settings))
            case "check-golden":
                aio.run(run_check_golden(settings))
            case "ask":
                aio.run(run_ask(settings, LLMSettings(), args.question, args.agent))
            case "chat":
                aio.run(run_chat(settings, LLMSettings()))
            case "eval":
                aio.run(run_eval(settings, LLMSettings(), args.agent, args.split, args.limit))
    except LLMError as err:
        raise SystemExit(f"Model call failed: {type(err).__name__}: {err}") from err
    except LoadError as err:
        raise SystemExit(f"Cannot set up the database: {err}") from err
    except psycopg.OperationalError as err:
        raise SystemExit(f"Cannot reach the database: {err}") from err
