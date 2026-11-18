"""Tests for the HTTP API (in-process app, in-memory fake Redis, fake researcher)."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable

import httpx
import pytest
from deep_research.api import create_app
from deep_research.config import DeepResearchSettings
from deep_research.jobs import JobStore
from deep_research.worker import ResearchFn, Worker
from fakeredis import FakeAsyncRedis

FakeResearch = Callable[..., ResearchFn]


@pytest.fixture
async def client(redis: FakeAsyncRedis) -> AsyncIterator[httpx.AsyncClient]:
    settings = DeepResearchSettings(_env_file=None, max_pending_jobs=2)  # type: ignore[call-arg]
    app = create_app(settings, redis=redis)
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as http,
    ):
        yield http


async def _submit(client: httpx.AsyncClient, question: str = "What is Artemis?") -> str:
    response = await client.post("/research", json={"question": question})
    assert response.status_code == 202
    job_id: str = response.json()["id"]
    return job_id


async def _run_one(store: JobStore, research: ResearchFn) -> None:
    """Let a worker process the next queued job."""
    await store.ensure_group()
    claimed = await store.next_job("test", block_ms=10, claim_idle_ms=60_000)
    assert claimed is not None
    await Worker(store, research, block_ms=10, cancel_poll_s=0.01).process(*claimed)


def _parse_sse(body: str) -> list[dict[str, str]]:
    """SSE frames as dicts of their fields; comment lines (keep-alives) are dropped."""
    frames = []
    for block in body.strip().split("\n\n"):
        fields = dict(
            line.split(": ", 1) for line in block.splitlines() if not line.startswith(":")
        )
        if fields:
            frames.append(fields)
    return frames


async def test_submit_returns_202_with_the_job_and_its_location(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post(
        "/research", json={"question": "  What is Artemis?  ", "researcher": "single-shot"}
    )

    assert response.status_code == 202
    body = response.json()
    assert response.headers["location"] == f"/research/{body['id']}"
    assert (body["status"], body["question"], body["researcher"]) == (
        "queued",
        "What is Artemis?",
        "single-shot",
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "Hi"},
        {"question": "x" * 501},
        {"question": "What is Artemis?", "researcher": "magic"},
        {"question": "What is Artemis?", "extra": 1},
    ],
)
async def test_invalid_requests_are_422(
    client: httpx.AsyncClient, payload: dict[str, object]
) -> None:
    assert (await client.post("/research", json=payload)).status_code == 422


async def test_full_queue_is_429_with_retry_after(client: httpx.AsyncClient) -> None:
    await _submit(client)
    await _submit(client)
    response = await client.post("/research", json={"question": "One too many?"})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "30"


async def test_get_returns_the_report_once_done(
    client: httpx.AsyncClient, redis: FakeAsyncRedis, fake_research: FakeResearch
) -> None:
    job_id = await _submit(client)
    assert (await client.get(f"/research/{job_id}")).json()["report"] is None

    await _run_one(JobStore(redis, max_pending=2, result_ttl_s=60), fake_research())

    body = (await client.get(f"/research/{job_id}")).json()
    assert body["status"] == "done"
    assert body["report"]["text"] == "agents answer [1]."


async def test_unknown_job_is_404_everywhere(client: httpx.AsyncClient) -> None:
    for method, path in [
        ("GET", "/research/nope"),
        ("GET", "/research/nope/events"),
        ("DELETE", "/research/nope"),
    ]:
        assert (await client.request(method, path)).status_code == 404


async def test_delete_cancels_a_queued_job_and_then_is_409(client: httpx.AsyncClient) -> None:
    job_id = await _submit(client)

    response = await client.delete(f"/research/{job_id}")
    assert (response.status_code, response.json()["status"]) == (202, "cancelled")
    assert (await client.delete(f"/research/{job_id}")).status_code == 409


async def test_events_replay_the_whole_run_and_end_after_the_final_event(
    client: httpx.AsyncClient, redis: FakeAsyncRedis, fake_research: FakeResearch
) -> None:
    job_id = await _submit(client)
    await _run_one(JobStore(redis, max_pending=2, result_ttl_s=60), fake_research())

    response = await client.get(f"/research/{job_id}/events")

    assert response.headers["content-type"].startswith("text/event-stream")
    frames = _parse_sse(response.text)
    assert [f["event"] for f in frames] == ["queued", "started", "progress", "progress", "done"]
    assert json.loads(frames[2]["data"])["node"] == "plan"
    assert all(f["id"] == json.loads(f["data"])["id"] for f in frames)


async def test_events_resume_after_last_event_id(
    client: httpx.AsyncClient, redis: FakeAsyncRedis, fake_research: FakeResearch
) -> None:
    job_id = await _submit(client)
    await _run_one(JobStore(redis, max_pending=2, result_ttl_s=60), fake_research())
    first = _parse_sse((await client.get(f"/research/{job_id}/events")).text)

    response = await client.get(
        f"/research/{job_id}/events", headers={"Last-Event-ID": first[2]["id"]}
    )
    assert [f["event"] for f in _parse_sse(response.text)] == ["progress", "done"]


async def test_bad_last_event_id_is_400(client: httpx.AsyncClient) -> None:
    job_id = await _submit(client)
    response = await client.get(f"/research/{job_id}/events", headers={"Last-Event-ID": "$"})
    assert response.status_code == 400


async def test_events_stream_live_while_the_worker_runs(
    client: httpx.AsyncClient, redis: FakeAsyncRedis, fake_research: FakeResearch
) -> None:
    job_id = await _submit(client)
    store = JobStore(redis, max_pending=2, result_ttl_s=60)

    async def worker_later() -> None:
        await asyncio.sleep(0.05)  # the stream is already open and waiting
        await _run_one(store, fake_research())

    response, _ = await asyncio.wait_for(
        asyncio.gather(client.get(f"/research/{job_id}/events"), worker_later()), timeout=5
    )
    assert [f["event"] for f in _parse_sse(response.text)][-1] == "done"


async def test_healthz(client: httpx.AsyncClient) -> None:
    response = await client.get("/healthz")
    assert (response.status_code, response.json()) == (200, {"status": "ok"})
