"""Tests for the Redis job store (on an in-memory fake Redis)."""

import pytest
from deep_research.api import HEARTBEAT_S
from deep_research.jobs import (
    GROUP,
    MAX_BLOCK_S,
    QUEUE,
    JobFinishedError,
    JobNotFoundError,
    JobStatus,
    JobStore,
    QueueFullError,
    connect,
    is_event_id,
)
from deep_research.report import Report
from deep_research.worker import Worker
from fakeredis import FakeAsyncRedis
from redis.exceptions import TimeoutError as RedisTimeoutError


async def _take(store: JobStore, consumer: str = "c1") -> tuple[str, str]:
    await store.ensure_group()
    claimed = await store.next_job(consumer, block_ms=10, claim_idle_ms=60_000)
    assert claimed is not None
    return claimed


async def test_submit_records_a_queued_job_with_an_event_and_a_queue_entry(
    store: JobStore, redis: FakeAsyncRedis
) -> None:
    job = await store.submit("What is Artemis?", "agents")

    assert (await store.get(job.id)) == job
    assert job.status is JobStatus.QUEUED
    assert [e.type for e in await store.events(job.id)] == ["queued"]
    assert await redis.xlen(QUEUE) == 1


async def test_submit_refuses_when_max_pending_jobs_are_queued(store: JobStore) -> None:
    for _ in range(3):
        await store.submit("Question?", "agents")
    with pytest.raises(QueueFullError):
        await store.submit("One too many?", "agents")


async def test_acked_jobs_free_their_queue_slot(store: JobStore) -> None:
    for _ in range(3):
        await store.submit("Question?", "agents")
    entry_id, _ = await _take(store)
    await store.ack(entry_id)
    await store.submit("Fits again?", "agents")


async def test_next_job_hands_out_each_entry_once(store: JobStore) -> None:
    job = await store.submit("Question?", "agents")
    _, job_id = await _take(store, "c1")
    assert job_id == job.id
    assert await store.next_job("c2", block_ms=10, claim_idle_ms=60_000) is None


async def test_next_job_reclaims_an_entry_a_dead_worker_left_pending(store: JobStore) -> None:
    job = await store.submit("Question?", "agents")
    entry_id, _ = await _take(store, "dead-worker")
    # Idle longer than claim_idle_ms (0 here): another consumer takes it over.
    assert await store.next_job("c2", block_ms=10, claim_idle_ms=0) == (entry_id, job.id)


async def test_start_marks_running_and_counts_the_attempt(store: JobStore) -> None:
    job = await store.submit("Question?", "agents")
    started = await store.start(job.id, max_attempts=2)

    assert started is not None
    assert (started.status, started.attempts) == (JobStatus.RUNNING, 1)
    assert started.started_at is not None
    assert [e.type for e in await store.events(job.id)] == ["queued", "started"]


async def test_start_gives_up_after_max_attempts(store: JobStore) -> None:
    job = await store.submit("Question?", "agents")
    await store.start(job.id, max_attempts=1)  # this worker then "dies"

    assert await store.start(job.id, max_attempts=1) is None
    failed = await store.get(job.id)
    assert failed is not None
    assert failed.status is JobStatus.FAILED
    assert failed.error is not None and "gave up after 1 attempt" in failed.error


async def test_start_skips_finished_and_missing_jobs(store: JobStore) -> None:
    job = await store.submit("Question?", "agents")
    await store.cancel(job.id)
    assert await store.start(job.id, max_attempts=2) is None
    assert await store.start("no-such-job", max_attempts=2) is None


async def test_cancel_of_a_queued_job_is_immediate_and_final(
    store: JobStore, redis: FakeAsyncRedis
) -> None:
    job = await store.submit("Question?", "agents")
    cancelled = await store.cancel(job.id)

    assert cancelled.status is JobStatus.CANCELLED
    events = await store.events(job.id)
    assert events[-1].type == "cancelled" and events[-1].final
    # Finished jobs and their events expire.
    assert 0 < await redis.ttl(f"research:job:{job.id}") <= 3600
    assert 0 < await redis.ttl(f"research:job:{job.id}:events") <= 3600


async def test_cancel_of_a_running_job_only_asks_the_worker(store: JobStore) -> None:
    job = await store.submit("Question?", "agents")
    await store.start(job.id, max_attempts=2)
    asked = await store.cancel(job.id)

    assert (asked.status, asked.cancel_requested) == (JobStatus.RUNNING, True)
    assert await store.cancel_requested(job.id)


async def test_cancel_errors(store: JobStore) -> None:
    with pytest.raises(JobNotFoundError):
        await store.cancel("no-such-job")
    job = await store.submit("Question?", "agents")
    await store.cancel(job.id)
    with pytest.raises(JobFinishedError):
        await store.cancel(job.id)


async def test_finish_stores_the_report_once(store: JobStore) -> None:
    job = await store.submit("Question?", "agents")
    await store.start(job.id, max_attempts=2)
    report = Report(question="Question?", text="Answer [1].")

    done = await store.finish(job.id, JobStatus.DONE, "ready", report=report)
    assert done is not None and done.report == report and done.finished_at is not None
    # A second final state (e.g. a late cancel) cannot overwrite the first.
    assert await store.finish(job.id, JobStatus.CANCELLED, "too late") is None
    assert (await store.get(job.id)).status is JobStatus.DONE  # type: ignore[union-attr]


async def test_requeue_hands_the_job_back_without_counting_the_attempt(
    store: JobStore, redis: FakeAsyncRedis
) -> None:
    job = await store.submit("Question?", "agents")
    entry_id, _ = await _take(store)
    await store.start(job.id, max_attempts=2)

    await store.requeue(entry_id, job.id)

    requeued = await store.get(job.id)
    assert requeued is not None
    assert (requeued.status, requeued.attempts) == (JobStatus.QUEUED, 0)
    assert [e.type for e in await store.events(job.id)][-1] == "requeued"
    # Old entry acked and deleted, a new one queued for the next worker.
    assert (await redis.xpending(QUEUE, GROUP))["pending"] == 0
    new_entry, job_id = await _take(store, "c2")
    assert (job_id, new_entry != entry_id) == (job.id, True)


async def test_requeued_job_with_a_cancel_request_is_cancelled_on_restart(
    store: JobStore,
) -> None:
    job = await store.submit("Question?", "agents")
    entry_id, _ = await _take(store)
    await store.start(job.id, max_attempts=2)
    await store.cancel(job.id)  # asks the worker...
    await store.requeue(entry_id, job.id)  # ...which shuts down instead

    assert await store.start(job.id, max_attempts=2) is None
    assert (await store.get(job.id)).status is JobStatus.CANCELLED  # type: ignore[union-attr]


async def test_events_after_an_id_returns_only_newer_ones(store: JobStore) -> None:
    job = await store.submit("Question?", "agents")
    first = (await store.events(job.id))[0]
    await store.add_event(job.id, "progress", "planned", node="plan")

    (newer,) = await store.events(job.id, after=first.id)
    assert (newer.type, newer.node, newer.message) == ("progress", "plan", "planned")
    assert await store.events(job.id, after=newer.id) == []


def test_is_event_id() -> None:
    assert is_event_id("1700000000000-0")
    assert not is_event_id("$") and not is_event_id("1-0; DROP")


async def test_requeue_of_a_finished_job_only_acks_it(
    store: JobStore, redis: FakeAsyncRedis
) -> None:
    job = await store.submit("Question?", "agents")
    entry_id, _ = await _take(store)
    await store.cancel(job.id)

    await store.requeue(entry_id, job.id)
    assert await redis.xlen(QUEUE) == 0


async def test_requeue_before_start_keeps_the_job_queued(store: JobStore) -> None:
    # Shutdown hit after the worker took the entry but before start() ran.
    job = await store.submit("Question?", "agents")
    entry_id, _ = await _take(store)

    await store.requeue(entry_id, job.id)
    assert (await store.get(job.id)) == job
    assert (await _take(store, "c2"))[1] == job.id


async def test_read_timeout_outlasts_every_blocking_read(store: JobStore) -> None:
    # redis-py's default 5 s socket timeout made idle blocking reads raise
    # TimeoutError against a real Redis (fakeredis never times out).
    client = connect("redis://localhost:6379/0")
    timeout = client.connection_pool.connection_kwargs["socket_timeout"]
    worker_block_s = Worker(store, research=None).block_ms / 1000  # type: ignore[arg-type]
    assert max(HEARTBEAT_S, worker_block_s) <= MAX_BLOCK_S < timeout
    await client.aclose()


async def test_blocking_reads_treat_a_redis_timeout_as_nothing_new(
    store: JobStore, redis: FakeAsyncRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = await store.submit("Question?", "agents")
    await store.ensure_group()

    async def timeout(*args: object, **kwargs: object) -> None:
        raise RedisTimeoutError("Timeout reading from redis")

    monkeypatch.setattr(redis, "xread", timeout)
    monkeypatch.setattr(redis, "xreadgroup", timeout)
    assert await store.events(job.id, block_ms=1000) == []
    assert await store.next_job("c1", block_ms=1000, claim_idle_ms=60_000) is None
    # A non-blocking read has no idle wait to excuse, so the error stands.
    with pytest.raises(RedisTimeoutError):
        await store.events(job.id)
