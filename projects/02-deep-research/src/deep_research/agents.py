"""The multi-agent researcher: a LangGraph graph of planner, researchers, writer and critic.

```
START -> plan -> research x N (parallel, one per sub-question) -> write
      -> critique -> (revise: write again | done: END)
```

- **plan**: one LLM call splits the question into at most
  ``max_sub_questions`` self-contained sub-questions.
- **research**: one branch per sub-question, fanned out with LangGraph's
  ``Send``, so the branches run concurrently. Each makes one LLM call to
  write keyword queries, runs them through ``SearchProvider`` in code and
  keeps the best passages. The LLM decides *what* to search; the code
  decides *that* it searches, so the flow is fixed and every run is
  bounded (no tool-calling loop that can wander).
- **write**: the shared writer (``writer.py``) turns all passages into a
  cited report, using the sub-questions as an outline.
- **critique**: a check in code (every sentence cited, no marker
  pointing nowhere) plus one LLM review of coverage. If either asks for
  changes and the revision budget allows, the writer runs again with the
  feedback.

Small models break structured output now and then, so every structured
step has a safe fallback (the question itself as the only sub-question,
the sub-question itself as the only query, the code check alone as the
review). A fallback costs quality, never the whole run.

Limits (``AgentLimits``) cap every loop and fan-out, so the number of
LLM calls per report is at most ``2 + max_sub_questions + 2 * max_revisions``
(plan, one query call per branch, the first draft, then one critique and
one rewrite per revision): 7 with the defaults.
"""

import operator
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, TypedDict

from aicore.client import LLMClient
from aicore.errors import StructuredOutputError
from aicore.models import LLMRequest, Message
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from pydantic import BaseModel, ConfigDict, Field

from deep_research.report import Report, markers, split_claims
from deep_research.search import Passage, SearchProvider
from deep_research.writer import build_writer_request

AGENT_PROMPT_VERSION = "v1"
"""Saved with eval results; bump it whenever a planner, query or critic prompt changes."""

PLANNER_PROMPT = """\
You plan research. Split the question into {n} or fewer sub-questions that \
together answer it. Each sub-question must make sense on its own: name the \
mission, rocket or organisation instead of writing "it" or "they". \
If the question is already simple, return it as the only sub-question."""

QUERY_PROMPT = """\
You write search queries for a keyword search engine over encyclopedia \
articles. Write {n} or fewer short queries (3 to 8 words each) that would \
find passages answering the sub-question. Use names, dates and technical \
terms likely to appear in the article text. Do not repeat the same query."""

CRITIC_PROMPT = """\
You review a research report. Check whether it answers every point listed. \
Approve it if it does. Otherwise list each missing or unanswered point \
as one short instruction to the writer (at most three)."""


class AgentLimits(BaseModel):
    """Caps on fan-out and loops, so cost and latency per report are bounded."""

    model_config = ConfigDict(frozen=True)

    max_sub_questions: int = Field(default=3, ge=1, le=6)
    queries_per_sub_question: int = Field(default=2, ge=1, le=5)
    passages_per_sub_question: int = Field(default=4, ge=1, le=10)
    max_revisions: int = Field(default=1, ge=0, le=3)


class Plan(BaseModel):
    """The JSON the planner must return."""

    model_config = ConfigDict(extra="forbid")

    sub_questions: list[str]


class Queries(BaseModel):
    """The JSON the query writer must return."""

    model_config = ConfigDict(extra="forbid")

    queries: list[str]


class Review(BaseModel):
    """The JSON the critic must return; ``reason`` first so it explains, then decides."""

    model_config = ConfigDict(extra="forbid")

    reason: str
    approved: bool
    fixes: list[str] = Field(default_factory=list)


class Finding(BaseModel):
    """What one researcher branch found for its sub-question."""

    model_config = ConfigDict(frozen=True)

    sub_question: str
    queries: list[str]
    passages: list[Passage]


class Critique(BaseModel):
    """The critic's verdict on one draft, combining the code check and the LLM review."""

    model_config = ConfigDict(frozen=True)

    approved: bool
    fixes: list[str]


class ProgressEvent(BaseModel):
    """One step of a research run, for showing progress to a user."""

    model_config = ConfigDict(frozen=True)

    node: str
    message: str


OnEvent = Callable[[ProgressEvent], Awaitable[None]]
"""Called with each finished step, e.g. to print it or publish it to a job's event stream."""


class ResearchState(TypedDict, total=False):
    """The graph's shared state. ``findings`` merges the parallel branches' outputs."""

    question: str
    sub_questions: list[str]
    findings: Annotated[list[Finding], operator.add]
    sources: list[Passage]
    report: str
    critique: Critique
    revisions: int


class BranchState(TypedDict):
    """The private input of one research branch."""

    sub_question: str


def merge_round_robin(result_lists: Sequence[Sequence[Passage]], limit: int) -> list[Passage]:
    """Take the 1st hit of every query, then the 2nd, ..., skipping repeats, up to ``limit``.

    BM25 scores from different queries are not on one scale, so sorting
    by score would let one query crowd out the others. Round-robin keeps
    every query's best passage.
    """
    merged: dict[str, Passage] = {}
    for rank in range(max((len(r) for r in result_lists), default=0)):
        for results in result_lists:
            if rank < len(results) and len(merged) < limit:
                merged.setdefault(results[rank].id, results[rank])
    return list(merged.values())


def review_citations(report: str, source_count: int) -> list[str]:
    """Fixes the code can demand without a model: uncited sentences and bad markers."""
    fixes: list[str] = []
    claims = split_claims(report)
    uncited = sum(not c.citations for c in claims)
    if uncited:
        fixes.append(
            f"{uncited} of {len(claims)} sentences have no source number. "
            "End every sentence with the numbers of its sources, like [1]."
        )
    bad = sorted({n for n in markers(report) if not 1 <= n <= source_count})
    if bad:
        fixes.append(f"Source numbers {bad} do not exist. Use only 1 to {source_count}.")
    return fixes


def _request(system: str, user: str, max_tokens: int) -> LLMRequest:
    """A deterministic request with one system and one user message."""
    return LLMRequest(
        messages=[Message(role="system", content=system), Message(role="user", content=user)],
        temperature=0.0,
        max_tokens=max_tokens,
    )


def _clean(items: Sequence[str], limit: int) -> list[str]:
    """Strip, drop empties and repeats (case-insensitive), keep at most ``limit``."""
    seen: dict[str, str] = {}
    for item in items:
        text = item.strip()
        if text and text.lower() not in seen:
            seen[text.lower()] = text
    return list(seen.values())[:limit]


class AgentResearcher:
    """``Researcher`` backed by the LangGraph graph described in the module docstring."""

    name = "agents"

    def __init__(
        self, search: SearchProvider, client: LLMClient, limits: AgentLimits | None = None
    ) -> None:
        """Build and compile the graph once; it is reused for every question."""
        self._search = search
        self._client = client
        self.limits = limits or AgentLimits()
        self.graph = self._build()

    async def __call__(self, question: str) -> Report:
        """Research ``question`` and return the final report.

        Raises:
            LLMError: If a model call fails after retries.
        """
        return await self.run(question)

    async def run(self, question: str, on_event: OnEvent | None = None) -> Report:
        """Research ``question``, awaiting ``on_event`` after each step finishes."""
        state: ResearchState = {"question": question, "findings": [], "revisions": 0}
        final: ResearchState = state
        async for mode, chunk in self.graph.astream(state, stream_mode=["updates", "values"]):
            if mode == "values":
                final = chunk  # type: ignore[assignment]
            elif on_event is not None:
                for node, update in chunk.items():  # type: ignore[union-attr]
                    await on_event(describe(node, update))
        return Report(
            question=question,
            text=final.get("report", ""),
            sources=final.get("sources", []),
            outline=final.get("sub_questions", []),
        )

    def _build(self) -> CompiledStateGraph:  # type: ignore[type-arg]
        """Wire the nodes and edges."""
        graph = StateGraph(ResearchState)
        graph.add_node("plan", self._plan)
        graph.add_node("research", self._research)
        graph.add_node("write", self._write)
        graph.add_node("critique", self._critique)
        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", self._fan_out, ["research"])
        graph.add_edge("research", "write")
        graph.add_conditional_edges("write", self._after_write, ["critique", END])
        graph.add_conditional_edges("critique", self._after_critique, ["write", END])
        return graph.compile()

    async def _plan(self, state: ResearchState) -> ResearchState:
        """Split the question into sub-questions (falls back to the question itself)."""
        question = state["question"]
        n = self.limits.max_sub_questions
        try:
            plan = await self._client.complete_structured(
                _request(PLANNER_PROMPT.format(n=n), f"Question: {question}", 300), Plan
            )
            sub_questions = _clean(plan.sub_questions, n)
        except StructuredOutputError:
            sub_questions = []
        return {"sub_questions": sub_questions or [question]}

    def _fan_out(self, state: ResearchState) -> list[Send]:
        """One research branch per sub-question; LangGraph runs them concurrently."""
        return [Send("research", {"sub_question": q}) for q in state["sub_questions"]]

    async def _research(self, state: BranchState) -> ResearchState:
        """Write queries for one sub-question, search each, keep the best passages."""
        sub_question = state["sub_question"]
        n = self.limits.queries_per_sub_question
        try:
            reply = await self._client.complete_structured(
                _request(QUERY_PROMPT.format(n=n), f"Sub-question: {sub_question}", 150), Queries
            )
            queries = _clean(reply.queries, n)
        except StructuredOutputError:
            queries = []
        queries = queries or [sub_question]
        k = self.limits.passages_per_sub_question
        results = [[h.passage for h in await self._search.search(q, k)] for q in queries]
        passages = merge_round_robin(results, k)
        finding = Finding(sub_question=sub_question, queries=queries, passages=passages)
        return {"findings": [finding]}

    async def _write(self, state: ResearchState) -> ResearchState:
        """Write the report, or rewrite it with the critic's fixes."""
        sources = list({p.id: p for f in state["findings"] for p in f.passages}.values())
        if not sources:
            return {"sources": [], "report": "No sources were found for this question."}
        critique = state.get("critique")
        revising = critique is not None and not critique.approved
        request = build_writer_request(
            state["question"],
            sources,
            outline=state["sub_questions"],
            draft=state["report"] if revising else None,
            feedback=critique.fixes if critique is not None else (),
        )
        response = await self._client.complete(request)
        revisions = state.get("revisions", 0) + (1 if revising else 0)
        return {"sources": sources, "report": response.text.strip(), "revisions": revisions}

    def _after_write(self, state: ResearchState) -> str:
        """Review the draft only if a revision could still follow."""
        if not state["sources"] or state["revisions"] >= self.limits.max_revisions:
            return END
        return "critique"

    async def _critique(self, state: ResearchState) -> ResearchState:
        """Code check for citations plus an LLM check for coverage (code check alone on failure)."""
        fixes = review_citations(state["report"], len(state["sources"]))
        points = "\n".join(f"- {q}" for q in state["sub_questions"])
        user = f"Points to answer:\n{points}\n\nReport:\n{state['report']}"
        try:
            review = await self._client.complete_structured(
                _request(CRITIC_PROMPT, user, 300), Review
            )
            if not review.approved:
                fixes.extend(_clean(review.fixes, 3) or [review.reason])
        except StructuredOutputError:
            pass
        return {"critique": Critique(approved=not fixes, fixes=fixes)}

    def _after_critique(self, state: ResearchState) -> str:
        """Revise when the critic asked for fixes (the write budget was checked before)."""
        return END if state["critique"].approved else "write"


def describe(node: str, update: ResearchState) -> ProgressEvent:
    """A one-line, human-readable summary of what ``node`` just did."""
    match node:
        case "plan":
            subs = update.get("sub_questions", [])
            message = f"planned {len(subs)} sub-question(s): " + " | ".join(subs)
        case "research":
            (finding,) = update.get("findings", [])
            message = (
                f"researched {finding.sub_question!r}: queries {finding.queries}, "
                f"{len(finding.passages)} passage(s)"
            )
        case "write":
            message = (
                f"wrote draft {update.get('revisions', 0) + 1} "
                f"from {len(update.get('sources', []))} source(s)"
            )
        case "critique":
            critique = update.get("critique")
            if critique is None or critique.approved:
                message = "critic approved the draft"
            else:
                message = "critic asked for fixes: " + " | ".join(critique.fixes)
        case _:
            message = "done"
    return ProgressEvent(node=node, message=message)
