"""Score generated answers on the golden dataset: refusal, citations, correctness.

Each golden question goes through the full ``answer_question`` pipeline,
one at a time so latencies are not distorted by concurrency. Three
things are measured, each in the cheapest way that is still honest:

- **Refusal**: unanswerable questions must be refused; answerable ones
  must not be (a false refusal). Pure bookkeeping, no model.
- **Citation precision**: a citation is supported when the cited chunk
  contains a labelled evidence quote (``metrics.is_relevant``, the same
  rule the retrieval eval uses). No model. It is a lower bound: a chunk
  that states the fact in words the label does not list counts as
  unsupported.
- **Correctness**: an LLM judge compares the answer with the golden
  ``reference_answer``. A judge can be wrong, so its verdicts are saved
  with a reason and spot-checked by hand.

The saved run records ``PROMPT_VERSION`` and ``JUDGE_VERSION``, so every
score is tied to the prompts that produced it.
"""

import math
import re
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

from aicore.client import LLMClient
from aicore.errors import StructuredOutputError
from aicore.models import LLMRequest, Message
from aicore.tracing import InMemoryTracer
from pydantic import BaseModel, ConfigDict, Field

from documind.ask import answer_question
from documind.generation import Answer, RefusalReason
from documind.golden import GoldenCase, Kind, Split
from documind.metrics import is_relevant
from documind.retrieval import Retriever

Verdict = Literal["correct", "partial", "incorrect"]

JUDGE_VERSION = "v1"
"""Saved with results; bump it whenever ``JUDGE_PROMPT`` changes."""

JUDGE_PROMPT = """\
You grade an answer to a question against a reference answer written by \
an expert. Judge only whether the facts match; ignore wording, length \
and source markers such as [1].

- "correct": the answer states the key facts of the reference and \
nothing that contradicts it. Extra detail is fine if it does not \
contradict the reference.
- "partial": the answer is right as far as it goes but leaves out a key \
fact the question asks for.
- "incorrect": the answer contradicts the reference, or answers a \
different question.

Give a one-sentence reason first, then the verdict."""

_MARKERS_RE = re.compile(r"\s*\[\d+\]")


class JudgeVerdict(BaseModel):
    """The JSON the judge must return. ``reason`` comes first so the
    model explains before it decides."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(description="One sentence comparing the answer to the reference.")
    verdict: Verdict


class AnswerCaseResult(BaseModel):
    """Everything measured for one golden question.

    ``citation_supported[i]`` says whether the i-th citation's chunk holds
    labelled evidence. ``evidence_cited`` counts evidence passages found
    in at least one cited chunk, out of ``evidence_total``. ``verdict`` is
    ``None`` when the judge did not run (refusals, unanswerable cases) or
    returned invalid JSON (``judge_error``).
    """

    model_config = ConfigDict(frozen=True)

    id: str
    kind: Kind
    question: str
    reference_answer: str
    answer: str
    draft: str = ""
    refusal_reason: RefusalReason | None = None
    blocked_by: str | None = None
    cited_chunk_ids: list[int] = Field(default_factory=list)
    citation_supported: list[bool] = Field(default_factory=list)
    invalid_citations: list[int] = Field(default_factory=list)
    evidence_total: int = 0
    evidence_cited: int = 0
    plain_text_reply: bool = False
    verdict: Verdict | None = None
    judge_reason: str = ""
    judge_error: bool = False
    latency_ms: float = Field(ge=0)
    cost_usd: float = Field(ge=0)

    @property
    def refused(self) -> bool:
        """Whether the system declined to answer."""
        return self.refusal_reason is not None


class AnswerReport(BaseModel):
    """All case results plus the counts the summary table needs."""

    model_config = ConfigDict(frozen=True)

    cases: list[AnswerCaseResult]
    judge_cost_usd: float = Field(default=0.0, ge=0)

    def of_kind(self, *kinds: Kind) -> list[AnswerCaseResult]:
        """Cases of the given kinds, in dataset order."""
        return [c for c in self.cases if c.kind in kinds]

    def answered(self, *kinds: Kind) -> list[AnswerCaseResult]:
        """Answerable cases of ``kinds`` that were not refused."""
        return [c for c in self.of_kind(*kinds) if not c.refused]

    def latency_percentile(self, pct: float) -> float | None:
        """Nearest-rank percentile of per-answer latency in ms, or ``None`` if no cases."""
        values = sorted(c.latency_ms for c in self.cases)
        if not values:
            return None
        return values[max(math.ceil(pct / 100 * len(values)) - 1, 0)]


class AnswerEvalRun(BaseModel):
    """One answer evaluation: the settings that produced it and the results."""

    model_config = ConfigDict(frozen=True)

    prompt_version: str
    judge_version: str
    model: str
    retriever: str
    split: Split | None
    context_chunks: int
    created_at: datetime
    report: AnswerReport

    def save(self, folder: Path) -> Path:
        """Write the run to ``folder/answers-{retriever}-{split}.json`` and return the path.

        A fixed name per retriever and split means a re-run overwrites the
        file and ``git diff`` shows which answers or verdicts changed.
        """
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"answers-{self.retriever}-{self.split or 'all'}.json"
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path


def score_citations(case: GoldenCase, answer: Answer) -> tuple[list[bool], int]:
    """Check each citation against the case's evidence, without a model.

    Returns:
        One flag per citation (does its chunk hold labelled evidence?) and
        the number of evidence passages found in at least one cited chunk.
    """
    cited = [answer.chunks[c.number - 1] for c in answer.citations]
    supported = [any(is_relevant(e, chunk) for e in case.evidence) for chunk in cited]
    covered = sum(any(is_relevant(e, chunk) for chunk in cited) for e in case.evidence)
    return supported, covered


def build_judge_request(case: GoldenCase, answer_text: str) -> LLMRequest:
    """The judge sees the question, the reference and the answer without markers."""
    candidate = _MARKERS_RE.sub("", answer_text).strip()
    user = (
        f"Question: {case.question}\n"
        f"Reference answer: {case.reference_answer}\n"
        f"Answer to grade: {candidate}"
    )
    return LLMRequest(
        messages=[
            Message(role="system", content=JUDGE_PROMPT),
            Message(role="user", content=user),
        ],
        temperature=0.0,
        max_tokens=200,
    )


async def judge_answer(
    case: GoldenCase, answer_text: str, client: LLMClient
) -> JudgeVerdict | None:
    """Grade ``answer_text`` against the reference; ``None`` if the judge's reply is invalid."""
    try:
        return await client.complete_structured(
            build_judge_request(case, answer_text), JudgeVerdict
        )
    except StructuredOutputError:
        return None


async def evaluate_answers(
    cases: Sequence[GoldenCase],
    retriever: Retriever,
    client: LLMClient,
    tracer: InMemoryTracer,
) -> AnswerReport:
    """Answer and score every case in ``cases``.

    ``tracer`` must be the one ``client`` reports to: it splits cost into
    answering (per case) and judging (in total).

    Raises:
        LLMError: If a retrieval or model call fails after retries.
    """
    results: list[AnswerCaseResult] = []
    judge_cost = 0.0
    for case in cases:
        before = len(tracer.records)
        start = time.perf_counter()
        answer = await answer_question(case.question, retriever, client)
        latency_ms = (time.perf_counter() - start) * 1000
        cost = _cost_since(tracer, before)

        supported, covered = score_citations(case, answer)
        verdict: JudgeVerdict | None = None
        judge_error = False
        if case.kind != "unanswerable" and not answer.refused:
            before = len(tracer.records)
            verdict = await judge_answer(case, answer.text, client)
            judge_cost += _cost_since(tracer, before)
            judge_error = verdict is None

        results.append(
            AnswerCaseResult(
                id=case.id,
                kind=case.kind,
                question=case.question,
                reference_answer=case.reference_answer,
                answer=answer.text,
                draft=answer.draft,
                refusal_reason=answer.refusal_reason,
                blocked_by=answer.blocked_by,
                cited_chunk_ids=[c.chunk_id for c in answer.citations],
                citation_supported=supported,
                invalid_citations=answer.invalid_citations,
                evidence_total=len(case.evidence),
                evidence_cited=covered,
                plain_text_reply=answer.plain_text_reply,
                verdict=verdict.verdict if verdict else None,
                judge_reason=verdict.reason if verdict else "",
                judge_error=judge_error,
                latency_ms=latency_ms,
                cost_usd=cost,
            )
        )
    return AnswerReport(cases=results, judge_cost_usd=judge_cost)


def _cost_since(tracer: InMemoryTracer, start: int) -> float:
    """Summed cost of the calls recorded after index ``start``."""
    return sum(r.cost_usd or 0.0 for r in tracer.records[start:])


def _ratio(num: int, den: int) -> str:
    """``0.83 (5/6)``, or ``-`` when there is nothing to divide by."""
    return f"{num / den:.2f} ({num}/{den})" if den else "-"


_METRIC_ROWS = (
    "Answered (not refused)",
    "Citation precision",
    "Answers with a supported citation",
    "Evidence passages cited",
    "Judge: correct (of answered)",
    "Judge: partial (of answered)",
    "Judge: incorrect (of answered)",
    "Correct end to end (of all)",
)


def _column(report: AnswerReport, kinds: tuple[Kind, ...]) -> list[str]:
    """One table column: every metric in ``_METRIC_ROWS`` for the cases of ``kinds``."""
    cases = report.of_kind(*kinds)
    answered = report.answered(*kinds)
    flags = [f for c in answered for f in c.citation_supported]
    return [
        _ratio(len(answered), len(cases)),
        _ratio(sum(flags), len(flags)),
        _ratio(sum(any(c.citation_supported) for c in answered), len(answered)),
        _ratio(sum(c.evidence_cited for c in answered), sum(c.evidence_total for c in answered)),
        *(
            _ratio(sum(c.verdict == v for c in answered), len(answered))
            for v in ("correct", "partial", "incorrect")
        ),
        _ratio(sum(c.verdict == "correct" for c in cases), len(cases)),
    ]


def format_answer_run(run: AnswerEvalRun) -> str:
    """Render ``run`` as Markdown: summary table, latency and cost, then every verdict."""
    report = run.report
    groups: list[tuple[str, tuple[Kind, ...]]] = [
        ("Answerable", ("factual", "multi-hop")),
        ("Factual", ("factual",)),
        ("Multi-hop", ("multi-hop",)),
    ]
    header = " | ".join(f"{name} ({len(report.of_kind(*kinds))})" for name, kinds in groups)
    columns = [_column(report, kinds) for _, kinds in groups]
    rows = [f"| Metric | {header} |", "|---" * (len(groups) + 1) + "|"]
    for i, label in enumerate(_METRIC_ROWS):
        rows.append(f"| {label} | " + " | ".join(col[i] for col in columns) + " |")

    unanswerable = report.of_kind("unanswerable")
    answerable = report.of_kind("factual", "multi-hop")
    refused_answerable = [c for c in answerable if c.refused]
    reasons = ", ".join(f"{c.id}: {c.refusal_reason}" for c in refused_answerable) or "none"
    counts = {
        "invalid citation markers": sum(len(c.invalid_citations) for c in report.cases),
        "plain-text (non-JSON) replies": sum(c.plain_text_reply for c in report.cases),
        "judge errors": sum(c.judge_error for c in report.cases),
    }
    n = len(report.cases)
    answer_cost = sum(c.cost_usd for c in report.cases)
    p50, p95 = report.latency_percentile(50), report.latency_percentile(95)

    lines = [
        f"Prompt {run.prompt_version}, judge {run.judge_version} | model: {run.model} | "
        f"retriever: {run.retriever} (top {run.context_chunks}) | "
        f"split: {run.split or 'all'} | {n} cases",
        "",
        *rows,
        "",
        f"Unanswerable refused: {_ratio(sum(c.refused for c in unanswerable), len(unanswerable))}",
        f"False refusals: {_ratio(len(refused_answerable), len(answerable))} ({reasons})",
        "; ".join(f"{k}: {v}" for k, v in counts.items()),
        "",
        f"Latency per answer: p50 {p50 or 0:.0f} ms, p95 {p95 or 0:.0f} ms",
        f"Cost: ${answer_cost / n if n else 0:.5f} per answer (${answer_cost:.4f} total), "
        f"judge ${report.judge_cost_usd:.4f}",
        "",
        "Judge verdicts (spot-check these):",
    ]
    for c in report.cases:
        if c.verdict is not None:
            lines.append(f"- {c.id} [{c.verdict}] {c.judge_reason}")
        elif c.judge_error:
            lines.append(f"- {c.id} [judge error]")
    return "\n".join(lines)
