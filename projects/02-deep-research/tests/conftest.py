"""Shared fixtures: an in-memory Redis and a job store on it."""

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence

import pytest
from deep_research.agents import OnEvent, ProgressEvent
from deep_research.jobs import JobStore
from deep_research.report import Report, ResearcherName
from deep_research.worker import ResearchFn
from fakeredis import FakeAsyncRedis


@pytest.fixture
async def redis() -> AsyncIterator[FakeAsyncRedis]:
    client = FakeAsyncRedis(decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture
def store(redis: FakeAsyncRedis) -> JobStore:
    return JobStore(redis, max_pending=3, result_ttl_s=3600)


FakeResearch = Callable[..., ResearchFn]


def _fake_research(
    nodes: Sequence[str] = ("plan", "write"),
    *,
    error: Exception | None = None,
    hang: bool = False,
) -> ResearchFn:
    """A researcher that reports ``nodes`` as steps, then fails, hangs or answers."""

    async def research(question: str, name: ResearcherName, on_event: OnEvent) -> Report:
        for node in nodes:
            await on_event(ProgressEvent(node=node, message=f"{node} step"))
        if error is not None:
            raise error
        if hang:
            await asyncio.Event().wait()
        return Report(question=question, text=f"{name} answer [1].")

    return research


@pytest.fixture
def fake_research() -> FakeResearch:
    """Factory for fake researchers (a fixture, so test modules need not import conftest)."""
    return _fake_research
