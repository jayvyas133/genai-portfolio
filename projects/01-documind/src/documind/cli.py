"""Command line for DocuMind: ingestion, evals, ``ask`` and ``serve`` (the HTTP API).

Run with ``uv run documind <command>``. Settings come from environment
variables or ``.env`` (``DOCUMIND_*`` for DocuMind, ``AICORE_*`` for the
models).
"""

import argparse
import logging
import time
from collections.abc import Sequence
from datetime import UTC, datetime

import psycopg
from aicore.config import LLMSettings
from aicore.errors import LLMError
from aicore.providers.factory import build_client
from aicore.tracing import CallRecord, InMemoryTracer, LogTracer

from documind import aio
from documind.answer_eval import (
    JUDGE_VERSION,
    AnswerEvalRun,
    evaluate_answers,
    format_answer_run,
)
from documind.api import create_app
from documind.ask import answer_question, refusal_message
from documind.config import DocumindSettings
from documind.corpus import load_documents
from documind.evaluate import EvalRun, evaluate_retrieval, format_run
from documind.generation import CONTEXT_CHUNKS, PROMPT_VERSION, Answer
from documind.golden import Split, load_golden
from documind.guards import answer_guards
from documind.pipeline import DimensionMismatchError, ingest
from documind.retrieval import RETRIEVER_NAMES, RetrieverName, build_retriever
from documind.store import PgChunkStore, connect, init_db

logger = logging.getLogger("documind")

_POSTGRES_HINT = "start it with: docker compose -f infra/docker-compose.yml up -d postgres"


async def run_init_db(settings: DocumindSettings) -> None:
    """Create the database and schema if missing (safe to re-run)."""
    created = await init_db(settings.database_url, settings.embedding_dimensions)
    print("Database created." if created else "Database already existed; schema is up to date.")


async def run_ingest(settings: DocumindSettings, llm_settings: LLMSettings) -> None:
    """Ingest every ``data/raw/*.json`` document and print a summary."""
    documents = load_documents(settings.data_dir / "raw")
    client = build_client(llm_settings, tracer=LogTracer())
    try:
        async with await connect(settings.database_url) as conn:
            store = PgChunkStore(conn)
            report = await ingest(documents, store, client, settings)
            stats = await store.stats()
    finally:
        await client.aclose()
    print(
        f"Ingested {report.documents} documents -> {report.chunks} chunks "
        f"({report.embedded} embedded, {report.reused} reused) "
        f"in {report.embed_calls} embed calls, {report.input_tokens} input tokens."
    )
    print(f"Store holds {stats.documents} documents and {stats.chunks} chunks.")


async def run_eval(
    settings: DocumindSettings,
    llm_settings: LLMSettings,
    name: RetrieverName,
    split: Split | None,
    k: int,
) -> None:
    """Score retriever ``name`` on the golden ``split``, print the table and save JSON."""
    cases = load_golden(settings.evals_dir / "golden.jsonl", split)
    client = build_client(llm_settings)
    try:
        async with await connect(settings.database_url) as conn:
            retriever = build_retriever(
                name, PgChunkStore(conn), client, settings.embedding_dimensions
            )
            report, skipped = await evaluate_retrieval(cases, retriever, k)
    finally:
        await client.aclose()
    run = EvalRun(
        retriever=name,
        split=split,
        k=k,
        embed_model=llm_settings.embed_model,
        rerank_model=llm_settings.rerank_model if name.endswith("-rerank") else None,
        chunk_words=settings.chunk_words,
        chunk_overlap_words=settings.chunk_overlap_words,
        created_at=datetime.now(UTC),
        skipped_unanswerable=skipped,
        report=report,
    )
    path = run.save(settings.evals_dir / "results")
    print(format_run(run))
    print()
    print(f"Saved {path}")


async def run_ask(
    settings: DocumindSettings, llm_settings: LLMSettings, question: str, name: RetrieverName
) -> None:
    """Answer ``question`` and print the answer, its sources, cost and latency."""
    tracer = InMemoryTracer()
    client = build_client(llm_settings, tracer=tracer, output_guards=answer_guards())
    start = time.perf_counter()
    try:
        async with await connect(settings.database_url) as conn:
            retriever = build_retriever(
                name, PgChunkStore(conn), client, settings.embedding_dimensions
            )
            answer = await answer_question(question, retriever, client)
    finally:
        await client.aclose()
    seconds = time.perf_counter() - start
    print(format_answer(answer, seconds=seconds, calls=tracer.records))


def format_answer(answer: Answer, *, seconds: float, calls: Sequence[CallRecord]) -> str:
    """Render ``answer`` for the terminal: text, numbered sources, then a stats line.

    The stats line shows total time, the time of each traced call
    (embed, rerank, complete) and their summed cost.
    """
    if answer.refused:
        lines = [f"No answer: {refusal_message(answer)}"]
    else:
        lines = [answer.text, "", "Sources:"]
        lines += [f"  [{c.number}] {c.document_title} > {c.section}" for c in answer.citations]
    cost = sum(c.cost_usd or 0.0 for c in calls)
    steps = ", ".join(f"{c.operation} {c.latency_ms / 1000:.1f} s" for c in calls)
    timing = f"{seconds:.1f} s ({steps})" if steps else f"{seconds:.1f} s"
    lines += ["", f"{timing}, ${cost:.4f}"]
    return "\n".join(lines)


async def run_eval_answers(
    settings: DocumindSettings, llm_settings: LLMSettings, name: RetrieverName, split: Split
) -> None:
    """Answer and score the golden ``split``, print the report and save JSON."""
    cases = load_golden(settings.evals_dir / "golden.jsonl", split)
    tracer = InMemoryTracer()
    client = build_client(llm_settings, tracer=tracer, output_guards=answer_guards())
    try:
        async with await connect(settings.database_url) as conn:
            retriever = build_retriever(
                name, PgChunkStore(conn), client, settings.embedding_dimensions
            )
            report = await evaluate_answers(cases, retriever, client, tracer)
    finally:
        await client.aclose()
    run = AnswerEvalRun(
        prompt_version=PROMPT_VERSION,
        judge_version=JUDGE_VERSION,
        model=llm_settings.model,
        retriever=name,
        split=split,
        context_chunks=CONTEXT_CHUNKS,
        created_at=datetime.now(UTC),
        report=report,
    )
    path = run.save(settings.evals_dir / "results")
    print(format_answer_run(run))
    print()
    print(f"Saved {path}")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the command and return the process exit code."""
    parser = argparse.ArgumentParser(prog="documind", description="DocuMind RAG service.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-db", help="Create the database and schema (safe to re-run).")
    commands.add_parser("ingest", help="Chunk, embed and store data/raw/*.json.")
    eval_cmd = commands.add_parser("eval", help="Score retrieval on the golden dataset.")
    eval_cmd.add_argument(
        "--split", choices=["dev", "test", "all"], default="dev", help="default: dev"
    )
    eval_cmd.add_argument(
        "--retriever",
        choices=RETRIEVER_NAMES,
        default="hybrid-rerank",
        help="default: hybrid-rerank",
    )
    eval_cmd.add_argument("--k", type=int, default=10, help="chunks to retrieve (default: 10)")
    ask_cmd = commands.add_parser("ask", help="Answer a question from the documents, with sources.")
    ask_cmd.add_argument("question", help="the question, in quotes")
    ask_cmd.add_argument(
        "--retriever",
        choices=RETRIEVER_NAMES,
        default="hybrid-rerank",
        help="default: hybrid-rerank",
    )
    answers_cmd = commands.add_parser(
        "eval-answers", help="Score answers: refusals, citations and an LLM judge."
    )
    answers_cmd.add_argument("--split", choices=["dev", "test"], default="dev", help="default: dev")
    answers_cmd.add_argument(
        "--retriever",
        choices=RETRIEVER_NAMES,
        default="hybrid-rerank",
        help="default: hybrid-rerank",
    )
    serve_cmd = commands.add_parser(
        "serve", help="Run the HTTP API on Flask's development server (gunicorn in Docker)."
    )
    serve_cmd.add_argument("--host", default="127.0.0.1", help="default: 127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8000, help="default: 8000")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is noise

    try:
        settings = DocumindSettings()
        if args.command == "serve":
            create_app().run(host=args.host, port=args.port)
            return 0
        if args.command == "init-db":
            coro = run_init_db(settings)
        elif args.command == "ingest":
            coro = run_ingest(settings, LLMSettings())
        elif args.command == "eval-answers":
            coro = run_eval_answers(settings, LLMSettings(), args.retriever, args.split)
        elif args.command == "ask":
            coro = run_ask(settings, LLMSettings(), args.question, args.retriever)
        else:
            split = None if args.split == "all" else args.split
            coro = run_eval(settings, LLMSettings(), args.retriever, split, args.k)
        aio.run(coro)
    except psycopg.OperationalError as exc:
        logger.error("Postgres not reachable (%s); %s", exc, _POSTGRES_HINT)
        return 1
    except (LLMError, DimensionMismatchError, FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0
