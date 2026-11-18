"""Tests for the research worker (fake researcher, in-memory fake Redis)."""

import asyncio
from collections.abc import Callable
from typing import Any

import pytest
from deep_research.jobs import QUEUE, Job, JobStatus, JobStore
from deep_research.worker import ResearchFn, Worker
from fakeredis import FakeAsyncRedis

FakeResearch = Callable[..., ResearchFn]


def _worker(store: JobStore, research: ResearchFn, timeout_s: float = 5.0) -> Worker:
    return Worker(store, research, job_timeout_s=timeout_s, block_ms=10, cancel_poll_s=0.01)


async def _queued(store: JobStore) -> tuple[Job, str]:
    """Submit a job and take its queue entry, as a worker loop would."""
    job = await store.submit("What is Artemis?", "agents")
    await store.ensure_group()
    claimed = await store.next_job("test", block_ms=10, claim_idle_ms=60_000)
    assert claimed is not None and claimed[1] == job.id
    return job, claimed[0]


async def _final(store: JobStore, job_id: str) -> Job:
    job = await store.get(job_id)
    assert job is not None and job.finished
    return job


async def test_successful_job_stores_the_report_publishes_steps_and_acks(
    store: JobStore, redis: FakeAsyncRedis, fake_research: FakeResearch
) -> None:
    job, entry_id = await _queued(store)
    await _worker(store, fake_research()).process(entry_id, job.id)

    done = await _final(store, job.id)
    assert done.status is JobStatus.DONE
    assert done.report is not None and done.report.text == "agents answer [1]."
    events = await store.events(job.id)
    assert [(e.type, e.node) for e in events] == [
        ("queued", None),
        ("started", None),
        ("progress", "plan"),
        ("progress", "write"),
        ("done", None),
    ]
    assert await redis.xlen(QUEUE) == 0


async def test_research_error_fails_the_job_not_the_worker(
    store: JobStore, fake_research: FakeResearch
) -> None:
    job, entry_id = await _queued(store)
    await _worker(store, fake_research(error=RuntimeError("model down"))).process(entry_id, job.id)

    failed = await _final(store, job.id)
    assert (failed.status, failed.error) == (JobStatus.FAILED, "RuntimeError: model down")


async def test_job_past_the_timeout_fails(store: JobStore, fake_research: FakeResearch) -> None:
    job, entry_id = await _queued(store)
    await _worker(store, fake_research(hang=True), timeout_s=0.05).process(entry_id, job.id)

    failed = await _final(store, job.id)
    assert failed.status is JobStatus.FAILED
    assert failed.error is not None and failed.error.startswith("timed out")


async def test_cancel_request_stops_a_running_job(
    store: JobStore, fake_research: FakeResearch
) -> None:
    job, entry_id = await _queued(store)
    run = asyncio.create_task(_worker(store, fake_research(hang=True)).process(entry_id, job.id))
    while (await store.get(job.id)).status is not JobStatus.RUNNING:  # type: ignore[union-attr]
        await asyncio.sleep(0.01)

    await store.cancel(job.id)
    await asyncio.wait_for(run, timeout=2)

    assert (await _final(store, job.id)).status is JobStatus.CANCELLED
    assert (await store.events(job.id))[-1].type == "cancelled"


async def test_shutdown_mid_job_requeues_it(store: JobStore, fake_research: FakeResearch) -> None:
    job, entry_id = await _queued(store)
    run = asyncio.create_task(_worker(store, fake_research(hang=True)).process(entry_id, job.id))
    while (await store.get(job.id)).status is not JobStatus.RUNNING:  # type: ignore[union-attr]
        await asyncio.sleep(0.01)

    run.cancel()  # what SIGTERM does to the worker
    await asyncio.wait({run})

    requeued = await store.get(job.id)
    assert requeued is not None and requeued.status is JobStatus.QUEUED
    claimed = await store.next_job("next-worker", block_ms=10, claim_idle_ms=60_000)
    assert claimed is not None and claimed[1] == job.id


async def test_job_cancelled_while_queued_is_skipped(
    store: JobStore, redis: FakeAsyncRedis, fake_research: FakeResearch
) -> None:
    job, entry_id = await _queued(store)
    await store.cancel(job.id)
    await _worker(store, fake_research(error=AssertionError("must not run"))).process(
        entry_id, job.id
    )

    assert (await _final(store, job.id)).status is JobStatus.CANCELLED
    assert await redis.xlen(QUEUE) == 0


async def test_run_loop_picks_up_submitted_jobs(
    store: JobStore, fake_research: FakeResearch, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real Redis waits up to block_ms when the queue is empty; fakeredis
    # returns at once without yielding, which would starve the event loop.
    next_job = store.next_job

    async def waiting_next_job(*args: Any) -> tuple[str, str] | None:
        claimed = await next_job(*args)
        if claimed is None:
            await asyncio.sleep(0.01)
        return claimed

    monkeypatch.setattr(store, "next_job", waiting_next_job)
    worker = Worker(store, fake_research(), concurrency=2, block_ms=10)
    running = asyncio.create_task(worker.run())
    jobs = [await store.submit(f"Question {i}?", "single-shot") for i in range(2)]
    try:
        for job in jobs:
            while not (await store.get(job.id)).finished:  # type: ignore[union-attr]
                await asyncio.sleep(0.01)
    finally:
        running.cancel()
        await asyncio.wait({running})
    for job in jobs:
        assert (await _final(store, job.id)).status is JobStatus.DONE
