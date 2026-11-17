"""Command line: ``documind init-db``, ``documind ingest`` and ``documind eval``.

Run with ``uv run documind <command>``. Settings come from environment
variables or ``.env`` (``DOCUMIND_*`` for DocuMind, ``AICORE_*`` for the
embedding model).
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

import psycopg
from aicore.config import LLMSettings
from aicore.errors import LLMError
from aicore.providers.factory import build_client
from aicore.tracing import LogTracer

from documind.config import DocumindSettings
from documind.corpus import load_documents
from documind.evaluate import EvalRun, evaluate_retrieval, format_run
from documind.golden import Split, load_golden
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


def _loop_factory() -> Callable[[], asyncio.AbstractEventLoop] | None:
    """Return a selector loop on Windows; psycopg async cannot use Proactor."""
    return asyncio.SelectorEventLoop if sys.platform == "win32" else None


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
        help="default: hybrid-rerank (chosen in M4, see ADR 0011)",
    )
    eval_cmd.add_argument("--k", type=int, default=10, help="chunks to retrieve (default: 10)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is noise

    try:
        settings = DocumindSettings()
        if args.command == "init-db":
            coro = run_init_db(settings)
        elif args.command == "ingest":
            coro = run_ingest(settings, LLMSettings())
        else:
            split = None if args.split == "all" else args.split
            coro = run_eval(settings, LLMSettings(), args.retriever, split, args.k)
        asyncio.run(coro, loop_factory=_loop_factory())
    except psycopg.OperationalError as exc:
        logger.error("Postgres not reachable (%s); %s", exc, _POSTGRES_HINT)
        return 1
    except (LLMError, DimensionMismatchError, FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0
