"""Run the golden questions through a retriever and score the results.

``evaluate_retrieval`` asks the retriever each answerable question, one
at a time so latencies are not distorted by concurrency, and scores the
ranked chunks with ``metrics.score_case``. ``EvalRun`` wraps the scores
with the settings that produced them, so a saved result says exactly
what was measured. ``format_run`` renders it as a Markdown table.
"""

import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from documind.golden import GoldenCase, Split
from documind.metrics import RetrievalReport, score_case
from documind.retrieval import Retriever

DEFAULT_KS = (1, 3, 5, 10)


class EvalRun(BaseModel):
    """One retrieval evaluation: what was run, on what, and the scores.

    ``skipped_unanswerable`` counts cases left out because they have no
    evidence to find (they are scored on refusal in M5).
    ``rerank_model`` is set only for retrievers that rerank.
    """

    model_config = ConfigDict(frozen=True)

    retriever: str
    split: Split | None
    k: int = Field(gt=0)
    embed_model: str
    rerank_model: str | None = None
    chunk_words: int
    chunk_overlap_words: int
    created_at: datetime
    skipped_unanswerable: int = Field(ge=0)
    report: RetrievalReport

    def save(self, folder: Path) -> Path:
        """Write the run to ``folder/{retriever}-{split}.json`` and return the path.

        The name is fixed per retriever and split, so a re-run overwrites
        the file and ``git diff`` shows exactly which cases changed.
        """
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self.retriever}-{self.split or 'all'}.json"
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path


async def evaluate_retrieval(
    cases: Sequence[GoldenCase], retriever: Retriever, k: int
) -> tuple[RetrievalReport, int]:
    """Retrieve ``k`` chunks for every answerable case and score them.

    Returns:
        The report and the number of unanswerable cases skipped.

    Raises:
        ValueError: If ``cases`` holds no answerable case.
    """
    answerable = [c for c in cases if c.evidence]
    if not answerable:
        msg = "no answerable cases to evaluate"
        raise ValueError(msg)
    scored = []
    for case in answerable:
        start = time.perf_counter()
        ranked = await retriever.retrieve(case.question, k)
        latency_ms = (time.perf_counter() - start) * 1000
        scored.append(score_case(case, ranked, latency_ms))
    return RetrievalReport(cases=scored), len(cases) - len(answerable)


def _cell(report: RetrievalReport | None, k: int | None, metric: str) -> str:
    """One table cell: the metric for ``report`` at ``k``, with a raw count for hit rate."""
    if report is None:
        return "-"
    n = len(report.cases)
    if metric == "hit" and k is not None:
        return f"{report.hit_rate(k):.2f} ({report.hits(k)}/{n})"
    if metric == "recall" and k is not None:
        return f"{report.recall(k):.2f}"
    return f"{report.mrr():.2f}"


def format_run(run: EvalRun, ks: Sequence[int] = DEFAULT_KS) -> str:
    """Render ``run`` as a Markdown table, overall and per question kind.

    Cut-offs in ``ks`` above ``run.k`` are dropped: they were not retrieved.
    """
    report = run.report
    columns = [
        ("All", report),
        ("Factual", report.only("factual")),
        ("Multi-hop", report.only("multi-hop")),
    ]
    header = " | ".join(f"{name} ({len(r.cases) if r else 0})" for name, r in columns)
    rows = [f"| Metric | {header} |", "|---" * (len(columns) + 1) + "|"]
    for metric, label in (("hit", "Hit rate"), ("recall", "Recall")):
        for k in (k for k in ks if k <= run.k):
            cells = " | ".join(_cell(r, k, metric) for _, r in columns)
            rows.append(f"| {label}@{k} | {cells} |")
    rows.append(f"| MRR@{run.k} | " + " | ".join(_cell(r, None, "mrr") for _, r in columns) + " |")

    p50, p95 = report.latency_percentile(50), report.latency_percentile(95)
    title = (
        f"Retriever: {run.retriever} | split: {run.split or 'all'} | "
        f"{len(report.cases)} cases ({run.skipped_unanswerable} unanswerable skipped) | "
        f"k: {run.k} | chunks: {run.chunk_words}/{run.chunk_overlap_words} words"
    )
    latency = (
        f"Latency per query (full retrieval): p50 {p50:.0f} ms, p95 {p95:.0f} ms"
        if p50 is not None and p95 is not None
        else "Latency: not measured"
    )
    return "\n".join([title, "", *rows, "", latency])
