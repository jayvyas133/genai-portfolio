"""Tests for the agent graph: planning, parallel research, writing, critique and limits."""

import json
from collections.abc import Callable

from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.models import LLMRequest, LLMResponse, Usage
from aicore.tracing import InMemoryTracer
from deep_research.agents import (
    CRITIC_PROMPT,
    PLANNER_PROMPT,
    QUERY_PROMPT,
    AgentLimits,
    AgentResearcher,
    ProgressEvent,
    merge_round_robin,
    review_citations,
)
from deep_research.search import FakeSearch, Passage, SearchHit
from deep_research.writer import WRITER_PROMPT

Reply = str | Callable[[LLMRequest], str]


class RoleProvider:
    """Fake model that answers by agent role (read from the system prompt).

    Research branches run concurrently, so the order of calls is not
    something a test should rely on; the role is. Each role has a list of
    replies used in turn (the last one repeats).
    """

    name = "fake"

    def __init__(self, **replies: list[Reply]) -> None:
        self.replies = replies
        self.calls: list[tuple[str, LLMRequest]] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        system = request.messages[-2].content
        roles = {
            "planner": PLANNER_PROMPT[:30],
            "queries": QUERY_PROMPT[:30],
            "writer": WRITER_PROMPT[:30],
            "critic": CRITIC_PROMPT[:30],
        }
        role = next(r for r, prefix in roles.items() if system.startswith(prefix))
        self.calls.append((role, request))
        options = self.replies[role]
        used = sum(1 for r, _ in self.calls if r == role)
        reply = options[min(used, len(options)) - 1]
        text = reply(request) if callable(reply) else reply
        return LLMResponse(text=text, model="fake", usage=Usage(input_tokens=1, output_tokens=1))

    def roles(self) -> list[str]:
        return [r for r, _ in self.calls]


def _passage(pid: str, title: str = "T") -> Passage:
    return Passage(id=pid, title=title, section="S", url="u", text=f"text of {pid}")


def _hits(*ids: str) -> list[SearchHit]:
    return [SearchHit(passage=_passage(i), score=1.0) for i in ids]


def _plan(*subs: str) -> str:
    return json.dumps({"sub_questions": list(subs)})


def _queries(request: LLMRequest) -> str:
    sub = request.messages[-1].content.removeprefix("Sub-question: ")
    return json.dumps({"queries": [f"{sub} q1", f"{sub} q2"]})


def _review(approved: bool, *fixes: str) -> str:
    return json.dumps({"reason": "r", "approved": approved, "fixes": list(fixes)})


def _researcher(
    provider: RoleProvider, search: FakeSearch, limits: AgentLimits | None = None
) -> AgentResearcher:
    client = LLMClient(provider, LLMSettings(_env_file=None), tracer=InMemoryTracer())
    return AgentResearcher(search, client, limits)


async def test_full_run_plans_researches_in_parallel_writes_and_gets_approved() -> None:
    provider = RoleProvider(
        planner=[_plan("Sub A", "Sub B")],
        queries=[_queries],
        writer=["A happened [1]. B happened [2]."],
        critic=[_review(True)],
    )
    search = FakeSearch(_hits("a#0", "b#0"))
    events: list[ProgressEvent] = []
    report = await _researcher(provider, search).run("Question?", on_event=events.append)

    assert report.text == "A happened [1]. B happened [2]."
    assert report.outline == ["Sub A", "Sub B"]
    # Both branches found the same passages; the writer gets each once.
    assert [p.id for p in report.sources] == ["a#0", "b#0"]
    assert sorted(search.queries) == ["Sub A q1", "Sub A q2", "Sub B q1", "Sub B q2"]
    assert provider.roles() == ["planner", "queries", "queries", "writer", "critic"]
    assert [e.node for e in events] == ["plan", "research", "research", "write", "critique"]
    assert events[-1].message == "critic approved the draft"
    writer_prompt = provider.calls[3][1].messages[-1].content
    assert "Cover these points:\n- Sub A\n- Sub B" in writer_prompt


async def test_uncited_draft_is_revised_once_with_the_feedback_then_stops() -> None:
    provider = RoleProvider(
        planner=[_plan("Sub A")],
        queries=[_queries],
        writer=["A happened. Also B.", "A happened [1]."],
        critic=[_review(False, "Mention B's date.")],
    )
    researcher = _researcher(provider, FakeSearch(_hits("a#0")))
    report = await researcher("Question?")

    assert report.text == "A happened [1]."
    # The budget allows one revision, so the rewrite is not reviewed again.
    assert provider.roles() == ["planner", "queries", "writer", "critic", "writer"]
    rewrite = provider.calls[-1][1].messages[-1].content
    assert "Your previous draft:\nA happened. Also B." in rewrite
    assert "2 of 2 sentences have no source number" in rewrite
    assert "- Mention B's date." in rewrite


async def test_zero_revisions_skips_the_critic() -> None:
    provider = RoleProvider(planner=[_plan("Sub A")], queries=[_queries], writer=["Done."])
    researcher = _researcher(provider, FakeSearch(_hits("a#0")), AgentLimits(max_revisions=0))
    await researcher("Question?")
    assert provider.roles() == ["planner", "queries", "writer"]


async def test_invalid_structured_replies_fall_back_instead_of_failing() -> None:
    provider = RoleProvider(
        planner=["not json"],
        queries=["not json"],
        writer=["Fine [1]."],
        critic=["not json"],
    )
    search = FakeSearch(_hits("a#0"))
    report = await _researcher(provider, search)("The question?")

    assert report.outline == ["The question?"]  # planner fallback: the question itself
    assert search.queries == ["The question?"]  # query fallback: the sub-question itself
    assert report.text == "Fine [1]."  # critic fallback: code check only, which passed
    assert provider.roles() == ["planner", "queries", "writer", "critic"]


async def test_plan_is_cleaned_and_capped_by_the_limits() -> None:
    provider = RoleProvider(
        planner=[_plan("A", " a ", "", "B", "C", "D")],
        queries=[_queries],
        writer=["x [1]."],
        critic=[_review(True)],
    )
    limits = AgentLimits(max_sub_questions=2, queries_per_sub_question=1)
    search = FakeSearch(_hits("a#0"))
    report = await _researcher(provider, search, limits)("Q?")
    assert report.outline == ["A", "B"]
    assert sorted(search.queries) == ["A q1", "B q1"]


async def test_no_sources_means_no_writer_call() -> None:
    provider = RoleProvider(planner=[_plan("A")], queries=[_queries])
    report = await _researcher(provider, FakeSearch([]))("Q?")
    assert report.sources == []
    assert report.text == "No sources were found for this question."
    assert provider.roles() == ["planner", "queries"]


def test_merge_round_robin_keeps_every_querys_best_hit_first() -> None:
    a = [_passage("a1"), _passage("a2"), _passage("a3")]
    b = [_passage("b1"), _passage("a2")]
    merged = merge_round_robin([a, b], limit=4)
    assert [p.id for p in merged] == ["a1", "b1", "a2", "a3"]
    assert [p.id for p in merge_round_robin([a, b], limit=2)] == ["a1", "b1"]
    assert merge_round_robin([], limit=3) == []


def test_review_citations_flags_uncited_sentences_and_unknown_numbers() -> None:
    assert review_citations("All good [1]. Yes [2].", source_count=2) == []
    fixes = review_citations("Cited [3]. Not cited.", source_count=2)
    assert fixes[0].startswith("1 of 2 sentences have no source number")
    assert fixes[1] == "Source numbers [3] do not exist. Use only 1 to 2."
