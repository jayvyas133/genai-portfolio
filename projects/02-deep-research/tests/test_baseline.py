"""Tests for the single-shot baseline researcher."""

from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.providers.fakeai import FakeProvider
from deep_research.baseline import SingleShotResearcher
from deep_research.search import FakeSearch, Passage, SearchHit
from deep_research.writer import build_writer_request


def _hit(pid: str, title: str, text: str) -> SearchHit:
    passage = Passage(id=pid, title=title, section="Intro", url="u", text=text)
    return SearchHit(passage=passage, score=1.0)


def test_writer_request_numbers_the_sources_and_asks_the_question() -> None:
    passages = [_hit("a#0", "A", "alpha").passage, _hit("b#0", "B", "beta").passage]
    request = build_writer_request("Why?", passages)
    user = request.messages[-1].content
    assert "[1] A - Intro\nalpha" in user
    assert "[2] B - Intro\nbeta" in user
    assert user.endswith("Question: Why?")
    assert request.temperature == 0.0


async def test_researcher_searches_once_with_the_question_and_keeps_the_sources() -> None:
    search = FakeSearch([_hit("a#0", "A", "alpha"), _hit("b#0", "B", "beta")])
    fake = FakeProvider(["  Alpha happened [1].  "])
    researcher = SingleShotResearcher(search, LLMClient(fake, LLMSettings(_env_file=None)), k=5)
    report = await researcher("What happened?")
    assert search.queries == ["What happened?"]
    assert report.text == "Alpha happened [1]."
    assert [p.id for p in report.sources] == ["a#0", "b#0"]
    assert len(fake.calls) == 1


async def test_researcher_skips_the_model_when_search_finds_nothing() -> None:
    fake = FakeProvider()
    researcher = SingleShotResearcher(
        FakeSearch([]), LLMClient(fake, LLMSettings(_env_file=None)), k=5
    )
    report = await researcher("q")
    assert report.sources == []
    assert fake.calls == []


def test_writer_request_adds_outline_draft_and_fixes_when_given() -> None:
    passage = _hit("a#0", "A", "alpha").passage
    user = (
        build_writer_request("Q?", [passage], outline=["Sub 1"], draft="Old.", feedback=["Fix"])
        .messages[-1]
        .content
    )
    assert "Cover these points:\n- Sub 1" in user
    assert "Your previous draft:\nOld.\n\nA reviewer asked for these fixes:\n- Fix" in user
