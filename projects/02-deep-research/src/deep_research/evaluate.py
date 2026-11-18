"""Score research reports on the golden dataset.

Any ``Researcher`` (the single-shot baseline now, the agent graph later)
runs on every case, one case at a time so latencies are not distorted by
concurrency. Each report is measured in the cheapest way that is still
honest:

- **Key-fact coverage** (LLM judge): which of the case's key facts the
  report states. One judge call per report grades all facts, each with a
  one-line reason, so a verdict can be audited fact by fact.
- **Key-fact retrieval** (code): which facts' quotes appear in a passage
  the system retrieved. Coverage much lower than retrieval means the
  writer dropped facts it had; low retrieval means search missed them.
- **Citations** (code): sentences without a marker and markers pointing
  at no source.
- **Citation faithfulness** (LLM judge): for up to ``max_claims`` cited
  sentences per report, does the cited passage actually support it?

``aicore.evals.run_eval`` is not used: its scorers are synchronous and
model-free by design, and both judges here are model calls. The saved
run records the model, the judge model and both prompt versions, so a
score is always tied to what produced it.
"""

import math
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from aicore.client import LLMClient
from aicore.errors import LLMError, StructuredOutputError
from aicore.models import LLMRequest, Message
from aicore.tracing import InMemoryTracer
from pydantic import BaseModel, ConfigDict, Field

from deep_research.golden import GoldenCase, KeyFact, Split, normalize
from deep_research.report import (
    Claim,
    Report,
    Researcher,
    invalid_markers,
    split_claims,
    strip_markers,
)
from deep_research.search import Passage

JUDGE_VERSION = "v1"
"""Saved with results; bump it whenever a judge prompt changes."""

FACTS_PROMPT = """\
You check which facts a research report states. For each numbered fact, \
decide whether the report states it, in any wording. A fact counts as \
stated only if every part of it is there (dates and numbers must match). \
Ignore source markers such as [1].

Reply with one verdict per fact, in order, each with a short reason."""

SUPPORT_PROMPT = """\
You check a sentence from a report against the source passages it cites. \
"supported" is true only if the passages state everything the sentence \
claims. If any part of the sentence is missing from the passages or \
contradicts them, it is false.

Give a one-sentence reason first, then the verdict."""


class FactVerdict(BaseModel):
    """The judge's verdict on one key fact; ``reason`` comes first so it explains, then decides."""

    model_config = ConfigDict(extra="forbid")

    fact: int = Field(description="The fact's number, starting at 1.")
    reason: str
    stated: bool


class FactVerdicts(BaseModel):
    """The JSON the fact judge must return."""

    model_config = ConfigDict(extra="forbid")

    verdicts: list[FactVerdict]


class SupportVerdict(BaseModel):
    """The JSON the support judge must return."""

    model_config = ConfigDict(extra="forbid")

    reason: str
    supported: bool


class CaseResult(BaseModel):
    """Everything measured for one golden question.

    ``facts_stated`` is empty when the fact judge failed
    (``fact_judge_error``). ``error`` names the exception class when the
    researcher itself failed; the case then has no scores.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    report: str = ""
    source_ids: list[str] = Field(default_factory=list)
    expected_titles: list[str]
    titles_found: list[str] = Field(default_factory=list)
    facts_retrieved: list[bool] = Field(default_factory=list)
    facts_stated: list[bool] = Field(default_factory=list)
    fact_reasons: list[str] = Field(default_factory=list)
    fact_judge_error: bool = False
    sentences: int = 0
    sentences_cited: int = 0
    invalid_markers: list[int] = Field(default_factory=list)
    claims_judged: int = 0
    claims_supported: int = 0
    support_judge_errors: int = 0
    unsupported: list[str] = Field(default_factory=list)
    latency_ms: float = Field(default=0.0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    error: str | None = None


class EvalRun(BaseModel):
    """One evaluation run: what produced it, and every case result."""

    model_config = ConfigDict(frozen=True)

    researcher: str
    model: str
    judge_model: str
    writer_prompt_version: str
    judge_version: str
    split: Split | None
    search_k: int
    max_claims: int
    created_at: datetime
    cases: list[CaseResult]
    judge_cost_usd: float = Field(default=0.0, ge=0)

    def save(self, folder: Path) -> Path:
        """Write the run to ``folder/{researcher}-{split}.json`` and return the path.

        A fixed name per researcher and split means a re-run overwrites the
        file and ``git diff`` shows which reports or verdicts changed.
        """
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self.researcher}-{self.split or 'all'}.json"
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path


def facts_retrieved(facts: Sequence[KeyFact], sources: Sequence[Passage]) -> list[bool]:
    """For each fact: does any retrieved passage of the right document hold its quote?"""
    return [
        any(s.title == f.title and normalize(f.quote) in normalize(s.text) for s in sources)
        for f in facts
    ]


def build_facts_request(facts: Sequence[KeyFact], report_text: str, model: str) -> LLMRequest:
    """The fact judge sees the numbered facts and the report without markers."""
    numbered = "\n".join(f"{i}. {f.fact}" for i, f in enumerate(facts, start=1))
    user = f"Facts:\n{numbered}\n\nReport:\n{strip_markers(report_text)}"
    return LLMRequest(
        messages=[
            Message(role="system", content=FACTS_PROMPT),
            Message(role="user", content=user),
        ],
        model=model,
        temperature=0.0,
        max_tokens=120 + 80 * len(facts),
    )


def build_support_request(claim: Claim, sources: Sequence[Passage], model: str) -> LLMRequest:
    """The support judge sees one sentence and the passages it cites."""
    cited = "\n\n".join(f"{sources[n - 1].title}: {sources[n - 1].text}" for n in claim.citations)
    user = f"Sentence: {strip_markers(claim.sentence)}\n\nCited passages:\n{cited}"
    return LLMRequest(
        messages=[
            Message(role="system", content=SUPPORT_PROMPT),
            Message(role="user", content=user),
        ],
        model=model,
        temperature=0.0,
        max_tokens=150,
    )


async def judge_facts(
    facts: Sequence[KeyFact], report_text: str, client: LLMClient, model: str
) -> list[FactVerdict] | None:
    """One verdict per fact, in order; ``None`` if the reply is invalid or the wrong length."""
    try:
        reply = await client.complete_structured(
            build_facts_request(facts, report_text, model), FactVerdicts
        )
    except StructuredOutputError:
        return None
    verdicts = sorted(reply.verdicts, key=lambda v: v.fact)
    if [v.fact for v in verdicts] != list(range(1, len(facts) + 1)):
        return None
    return verdicts


async def judge_support(
    claim: Claim, sources: Sequence[Passage], client: LLMClient, model: str
) -> SupportVerdict | None:
    """Whether the cited passages support ``claim``; ``None`` if the reply is invalid."""
    try:
        return await client.complete_structured(
            build_support_request(claim, sources, model), SupportVerdict
        )
    except StructuredOutputError:
        return None


async def evaluate(
    cases: Sequence[GoldenCase],
    researcher: Researcher,
    client: LLMClient,
    tracer: InMemoryTracer,
    *,
    judge_model: str,
    max_claims: int = 8,
) -> tuple[list[CaseResult], float]:
    """Run ``researcher`` on every case and score each report.

    ``tracer`` must be the one ``client`` reports to: it splits cost and
    call counts into research (per case) and judging (in total).

    Returns:
        The case results and the total judge cost in US dollars.

    Raises:
        LLMError: If a judge call fails after retries (a researcher
            failure is recorded on its case instead).
    """
    results: list[CaseResult] = []
    judge_cost = 0.0
    for case in cases:
        before = len(tracer.records)
        start = time.perf_counter()
        try:
            report = await researcher(case.question)
        except LLMError as err:
            results.append(
                CaseResult(
                    id=case.id,
                    question=case.question,
                    expected_titles=case.expected_titles,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    llm_calls=len(tracer.records) - before,
                    error=type(err).__name__,
                )
            )
            continue
        latency_ms = (time.perf_counter() - start) * 1000
        research_records = tracer.records[before:]

        before = len(tracer.records)
        result = await _score(case, report, client, judge_model, max_claims)
        judge_cost += sum(r.cost_usd or 0.0 for r in tracer.records[before:])
        results.append(
            result.model_copy(
                update={
                    "latency_ms": latency_ms,
                    "cost_usd": sum(r.cost_usd or 0.0 for r in research_records),
                    "llm_calls": len(research_records),
                }
            )
        )
    return results, judge_cost


async def _score(
    case: GoldenCase, report: Report, client: LLMClient, judge_model: str, max_claims: int
) -> CaseResult:
    """All scores for one report (latency, cost and calls are filled in by the caller)."""
    verdicts = await judge_facts(case.key_facts, report.text, client, judge_model)
    claims = split_claims(report.text)
    bad = set(invalid_markers(report))
    checkable = [c for c in claims if c.citations and not bad.intersection(c.citations)]

    judged = supported = errors = 0
    unsupported: list[str] = []
    for claim in checkable[:max_claims]:
        verdict = await judge_support(claim, report.sources, client, judge_model)
        if verdict is None:
            errors += 1
            continue
        judged += 1
        if verdict.supported:
            supported += 1
        else:
            unsupported.append(f"{claim.sentence} -- {verdict.reason}")

    found = {s.title for s in report.sources}
    return CaseResult(
        id=case.id,
        question=case.question,
        report=report.text,
        source_ids=[s.id for s in report.sources],
        expected_titles=case.expected_titles,
        titles_found=[t for t in case.expected_titles if t in found],
        facts_retrieved=facts_retrieved(case.key_facts, report.sources),
        facts_stated=[v.stated for v in verdicts] if verdicts else [],
        fact_reasons=[v.reason for v in verdicts] if verdicts else [],
        fact_judge_error=verdicts is None,
        sentences=len(claims),
        sentences_cited=sum(bool(c.citations) for c in claims),
        invalid_markers=sorted(bad),
        claims_judged=judged,
        claims_supported=supported,
        support_judge_errors=errors,
        unsupported=unsupported,
    )


def _ratio(num: int, den: int) -> str:
    """``0.83 (5/6)``, or ``-`` when there is nothing to divide by."""
    return f"{num / den:.2f} ({num}/{den})" if den else "-"


def _percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile (an observed value), or 0.0 for no values."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[max(math.ceil(pct / 100 * len(ordered)) - 1, 0)]


def format_run(run: EvalRun) -> str:
    """Render ``run`` as Markdown: summary table, latency and cost, then per-case detail."""
    ok = [c for c in run.cases if c.error is None]
    judged = [c for c in ok if not c.fact_judge_error]
    rows = [
        ("Key facts stated (judge)", _ratio(*_count(judged, "facts_stated"))),
        ("Key facts retrieved (quote in a source)", _ratio(*_count(ok, "facts_retrieved"))),
        (
            "Expected documents retrieved",
            _ratio(sum(len(c.titles_found) for c in ok), sum(len(c.expected_titles) for c in ok)),
        ),
        (
            "Sentences with a citation",
            _ratio(sum(c.sentences_cited for c in ok), sum(c.sentences for c in ok)),
        ),
        (
            "Cited sentences supported (judge)",
            _ratio(sum(c.claims_supported for c in ok), sum(c.claims_judged for c in ok)),
        ),
    ]
    n = len(run.cases)
    research_cost = sum(c.cost_usd for c in run.cases)
    latencies = [c.latency_ms for c in ok]
    counts = {
        "researcher errors": n - len(ok),
        "fact judge errors": sum(c.fact_judge_error for c in ok),
        "support judge errors": sum(c.support_judge_errors for c in ok),
        "invalid citation markers": sum(len(c.invalid_markers) for c in ok),
    }
    lines = [
        f"Researcher: {run.researcher} | model: {run.model} | judge: {run.judge_model} | "
        f"writer {run.writer_prompt_version}, judge {run.judge_version} | "
        f"split: {run.split or 'all'} | {n} cases | search k={run.search_k}",
        "",
        "| Metric | Value |",
        "|---|---|",
        *(f"| {label} | {value} |" for label, value in rows),
        "",
        "; ".join(f"{k}: {v}" for k, v in counts.items()),
        f"Latency per report: p50 {_percentile(latencies, 50) / 1000:.1f} s, "
        f"p95 {_percentile(latencies, 95) / 1000:.1f} s",
        f"LLM calls per report: {sum(c.llm_calls for c in run.cases) / n if n else 0:.1f}",
        f"Cost: ${research_cost / n if n else 0:.5f} per report "
        f"(${research_cost:.4f} total), judge ${run.judge_cost_usd:.4f}",
        "",
        "Per case (spot-check the judge):",
    ]
    for c in run.cases:
        if c.error:
            lines.append(f"- {c.id}: error {c.error}")
            continue
        stated = "judge error" if c.fact_judge_error else f"{sum(c.facts_stated)}"
        lines.append(
            f"- {c.id}: facts stated {stated}/{len(c.facts_retrieved)}, "
            f"retrieved {sum(c.facts_retrieved)}/{len(c.facts_retrieved)}, "
            f"supported {c.claims_supported}/{c.claims_judged}"
        )
    return "\n".join(lines)


def _count(cases: Sequence[CaseResult], field: str) -> tuple[int, int]:
    """Sum of true flags and total flags in list field ``field`` across ``cases``."""
    flags = [flag for c in cases for flag in getattr(c, field)]
    return sum(flags), len(flags)
