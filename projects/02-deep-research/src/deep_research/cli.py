"""Command line for Deep Research.

Run with ``uv run deep-research <command>``. Settings come from
environment variables or ``.env`` (``DEEP_RESEARCH_*`` for this project,
``AICORE_*`` for the models).
"""

import argparse
import asyncio
import logging
import time
from collections.abc import Sequence
from datetime import UTC, datetime

from aicore.config import LLMSettings
from aicore.errors import LLMError
from aicore.providers.factory import build_client
from aicore.tracing import InMemoryTracer

from deep_research.baseline import WRITER_PROMPT_VERSION, SingleShotResearcher
from deep_research.config import DeepResearchSettings
from deep_research.corpus import TITLES, fetch_documents, load_documents
from deep_research.evaluate import JUDGE_VERSION, EvalRun, evaluate, format_run
from deep_research.golden import Split, check_quotes, load_golden
from deep_research.search import CorpusSearch

logger = logging.getLogger("deep_research")


def _search(settings: DeepResearchSettings) -> CorpusSearch:
    """Load the corpus and build the search index."""
    return CorpusSearch(load_documents(settings.corpus_dir), settings.passage_words)


async def run_fetch_corpus(settings: DeepResearchSettings) -> None:
    """Download any corpus articles not yet on disk."""
    paths = await fetch_documents(TITLES, settings.corpus_dir)
    print(f"Corpus has {len(paths)} documents in {settings.corpus_dir}")


async def run_search(settings: DeepResearchSettings, query: str, k: int) -> None:
    """Print the top ``k`` passages for ``query``."""
    for hit in await _search(settings).search(query, k):
        p = hit.passage
        print(f"{hit.score:6.2f}  {p.id}  ({p.title} - {p.section})")
        print(f"        {p.text[:200]}")


async def run_ask(settings: DeepResearchSettings, llm: LLMSettings, question: str) -> None:
    """Research ``question`` with the baseline and print the report and its sources."""
    tracer = InMemoryTracer()
    client = build_client(llm, tracer=tracer)
    start = time.perf_counter()
    try:
        report = await SingleShotResearcher(_search(settings), client, settings.search_k)(question)
    finally:
        await client.aclose()
    print(report.text)
    print()
    for i, p in enumerate(report.sources, start=1):
        print(f"[{i}] {p.title} - {p.section} ({p.url})")
    cost = sum(r.cost_usd or 0.0 for r in tracer.records)
    print(f"\n{time.perf_counter() - start:.1f} s, {len(tracer.records)} LLM call(s), ${cost:.5f}")


async def run_eval(
    settings: DeepResearchSettings,
    llm: LLMSettings,
    split: Split | None,
    limit: int | None,
    judge_model: str | None,
    max_claims: int,
) -> None:
    """Evaluate the baseline on the golden ``split``, print the table and save JSON."""
    cases = load_golden(settings.evals_dir / "golden.jsonl", split)[:limit]
    documents = load_documents(settings.corpus_dir)
    if problems := check_quotes(cases, documents):
        raise SystemExit("Golden dataset does not match the corpus:\n" + "\n".join(problems))
    judge = judge_model or llm.model
    tracer = InMemoryTracer()
    client = build_client(llm, tracer=tracer)
    researcher = SingleShotResearcher(
        CorpusSearch(documents, settings.passage_words), client, settings.search_k
    )
    try:
        results, judge_cost = await evaluate(
            cases, researcher, client, tracer, judge_model=judge, max_claims=max_claims
        )
    finally:
        await client.aclose()
    run = EvalRun(
        researcher=researcher.name,
        model=llm.model,
        judge_model=judge,
        writer_prompt_version=WRITER_PROMPT_VERSION,
        judge_version=JUDGE_VERSION,
        split=split,
        search_k=settings.search_k,
        max_claims=max_claims,
        created_at=datetime.now(UTC),
        cases=results,
        judge_cost_usd=judge_cost,
    )
    path = run.save(settings.evals_dir / "results")
    print(format_run(run))
    print(f"\nSaved {path}")


def build_parser() -> argparse.ArgumentParser:
    """The ``deep-research`` argument parser."""
    parser = argparse.ArgumentParser(prog="deep-research", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch-corpus", help="download missing corpus articles from Wikipedia")

    search = sub.add_parser("search", help="show the top passages for a query")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=5)

    ask = sub.add_parser("ask", help="research a question and print a cited report")
    ask.add_argument("question")

    ev = sub.add_parser("eval", help="score the researcher on the golden dataset")
    ev.add_argument("--split", choices=["dev", "test"], default=None)
    ev.add_argument("--limit", type=int, default=None, help="only the first N cases")
    ev.add_argument("--judge-model", default=None, help="defaults to AICORE_MODEL")
    ev.add_argument(
        "--max-claims", type=int, default=8, help="cited sentences judged per report (0: none)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for ``uv run deep-research``."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    settings = DeepResearchSettings()
    try:
        match args.command:
            case "fetch-corpus":
                asyncio.run(run_fetch_corpus(settings))
            case "search":
                asyncio.run(run_search(settings, args.query, args.k))
            case "ask":
                asyncio.run(run_ask(settings, LLMSettings(), args.question))
            case "eval":
                asyncio.run(
                    run_eval(
                        settings,
                        LLMSettings(),
                        args.split,
                        args.limit,
                        args.judge_model,
                        args.max_claims,
                    )
                )
    except LLMError as err:
        raise SystemExit(f"Model call failed: {type(err).__name__}: {err}") from err
