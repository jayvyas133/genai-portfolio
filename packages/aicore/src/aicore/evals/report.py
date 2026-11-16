"""Render an ``EvalReport`` as Markdown for READMEs and pull requests."""

from aicore.evals.models import CaseResult, EvalReport


def to_markdown(report: EvalReport, *, max_failures: int = 10) -> str:
    """Return a summary table, a per-scorer table and the failed cases.

    At most ``max_failures`` failed cases are listed, so a bad run
    still gives a readable report.
    """
    passed = sum(r.passed for r in report.results)
    p50 = report.latency_percentile(50)
    p95 = report.latency_percentile(95)
    lines = [
        f"### Eval: {report.name}",
        "",
        f"Run started {report.started_at.isoformat(timespec='minutes')}.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Cases | {len(report.results)} |",
        f"| Passed | {passed} ({report.pass_rate:.1%}) |",
        f"| Target errors | {report.error_count} |",
        f"| Latency p50 / p95 | {p50:.2f} ms / {p95:.2f} ms |",
        f"| LLM calls | {report.llm_calls} |",
        f"| Total cost | {_usd(report.total_cost_usd)} |",
    ]

    rates = report.scorer_pass_rates()
    if rates:
        lines += ["", "| Scorer | Pass rate |", "|---|---|"]
        lines += [f"| {_cell(name)} | {rate:.1%} |" for name, rate in rates.items()]

    failures = [r for r in report.results if not r.passed]
    if failures:
        shown = failures[:max_failures]
        lines += ["", f"Failed cases ({len(shown)} of {len(failures)} shown):", ""]
        lines += ["| Case | Problem |", "|---|---|"]
        lines += [f"| {_cell(r.case_id)} | {_cell(_problem(r))} |" for r in shown]

    return "\n".join(lines) + "\n"


def _problem(result: CaseResult) -> str:
    """Describe why ``result`` failed in one line."""
    if result.error is not None:
        return f"error: {result.error}"
    return "; ".join(f"{s.name}: {s.reason or 'failed'}" for s in result.scores if not s.passed)


def _usd(cost: float | None) -> str:
    """Format a cost; ``None`` means no priced call was made."""
    return "unknown" if cost is None else f"USD {cost:.6f}"


def _cell(text: str) -> str:
    """Make ``text`` safe inside a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")
