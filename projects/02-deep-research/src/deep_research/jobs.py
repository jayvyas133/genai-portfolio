"""Research jobs in Redis: the job queue, job records and progress events.

A research run takes minutes, far longer than an HTTP request should
stay open, so the API only *submits* a job and a separate worker process
runs it (see ``worker.py``). Redis holds three things per deployment:

- ``research:queue``, a Redis Stream used as the job queue and read by
  the ``workers`` consumer group. A worker acknowledges an entry
  (``XACK``) only once the job has reached a final state, so if a worker
  dies mid-job the entry stays pending and another worker takes it over
  (``XAUTOCLAIM``) when it has been idle longer than the job timeout.
  Delivery is therefore at-least-once; ``Job.attempts`` caps the
  redeliveries, so a job that keeps killing workers fails instead of
  looping forever.
- ``research:job:<id>``, the job record as one JSON document. Every
  change is a read-check-write under ``WATCH`` (optimistic locking): if
  the API cancels while the worker finishes, one of the two transactions
  is retried on the new state instead of silently overwriting it.
- ``research:job:<id>:events``, a Redis Stream of progress events. Stream
  entry ids are ordered, so they double as Server-Sent Events ids: a
  client that reconnects with ``Last-Event-ID`` reads on from there.

Finished jobs and their events expire after ``result_ttl_s``.
"""

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict
from redis.asyncio import Redis
from redis.asyncio.client import Pipeline
from redis.exceptions import ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError

from deep_research.report import Report, ResearcherName

QUEUE = "research:queue"
GROUP = "workers"

_EVENT_ID_RE = re.compile(r"^\d+-\d+$")

MAX_BLOCK_S = 15.0
"""Longest a blocking read (XREAD / XREADGROUP with ``block``) may wait."""

READ_TIMEOUT_S = MAX_BLOCK_S + 15.0
"""Socket read timeout. It must exceed every blocking read, or redis-py raises
``TimeoutError`` on an idle wait; its own default (5 s) is too short. Even
so, a stalled host can stretch a wait past it, so blocking reads treat a
timeout as "nothing new yet" (see ``events`` and ``next_job``)."""


class JobStatus(StrEnum):
    """Where a job is in its life: queued -> running -> done | failed | cancelled."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


FINAL_STATUSES = frozenset({JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED})

EventType = Literal["queued", "started", "progress", "requeued", "done", "failed", "cancelled"]
"""``done``, ``failed`` and ``cancelled`` are final: nothing follows them."""


class Job(BaseModel):
    """One research request and, once finished, its outcome."""

    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    researcher: ResearcherName
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    # Times a worker has started it; more than one means a worker died.
    attempts: int = 0
    # Set by the API while the job runs; the worker polls it and stops.
    cancel_requested: bool = False
    error: str | None = None
    report: Report | None = None

    @property
    def finished(self) -> bool:
        """True once the job is done, failed or cancelled."""
        return self.status in FINAL_STATUSES


class JobEvent(BaseModel):
    """One entry of a job's event stream."""

    model_config = ConfigDict(frozen=True)

    id: str
    type: EventType
    message: str
    # The graph node that produced a ``progress`` event.
    node: str | None = None

    @property
    def final(self) -> bool:
        """True for the last event a job ever gets."""
        return self.type in FINAL_STATUSES


class QueueFullError(Exception):
    """Too many jobs are queued or running; the caller should retry later."""


class JobNotFoundError(Exception):
    """No such job (never existed, or expired)."""


class JobFinishedError(Exception):
    """The job already finished, so it cannot be cancelled."""


def connect(url: str) -> Redis:
    """A Redis client for the job store: text replies and a read timeout above ``MAX_BLOCK_S``."""
    return Redis.from_url(url, decode_responses=True, socket_timeout=READ_TIMEOUT_S)


def is_event_id(value: str) -> bool:
    """True if ``value`` looks like a Redis Stream entry id (``<ms>-<seq>``)."""
    return _EVENT_ID_RE.match(value) is not None


def _job_key(job_id: str) -> str:
    return f"research:job:{job_id}"


def _events_key(job_id: str) -> str:
    return f"research:job:{job_id}:events"


def _now() -> datetime:
    return datetime.now(UTC)


def _add_event(
    pipe: Pipeline, job_id: str, type_: EventType, message: str, node: str | None = None
) -> None:
    """Queue an XADD of one event on ``pipe`` (Redis cannot store None, so no node field)."""
    fields = {"type": type_, "message": message}
    if node is not None:
        fields["node"] = node
    pipe.xadd(_events_key(job_id), fields)  # type: ignore[arg-type]


def _begin(pipe: Pipeline) -> None:
    """Start the MULTI block of a WATCHed transaction (redis-py leaves ``multi`` untyped)."""
    pipe.multi()  # type: ignore[no-untyped-call]


Entry = tuple[str, dict[str, str]]
"""One stream entry as redis-py returns it with ``decode_responses``: id and fields."""


def _entries(reply: Any) -> list[Entry]:
    """The entries of an XREAD / XREADGROUP reply (``[[stream, [entry, ...]], ...]``)."""
    return [cast(Entry, tuple(entry)) for _, entries in reply or [] for entry in entries]


async def _read(pipe: Pipeline, job_id: str) -> Job | None:
    """Read the job inside a WATCHed transaction (the pipeline is in immediate mode)."""
    raw = await pipe.get(_job_key(job_id))
    return None if raw is None else Job.model_validate_json(raw)


class JobStore:
    """All Redis access for jobs, shared by the API and the worker."""

    def __init__(self, redis: Redis, *, max_pending: int, result_ttl_s: int) -> None:
        """Wrap ``redis``, which must use ``decode_responses=True``."""
        self._redis = redis
        self._max_pending = max_pending
        self._result_ttl_s = result_ttl_s

    async def ping(self) -> bool:
        """True if Redis answers."""
        return bool(await self._redis.ping())

    # ----- API side -------------------------------------------------------

    async def submit(self, question: str, researcher: ResearcherName) -> Job:
        """Record a new job and put it in the queue.

        Raises:
            QueueFullError: If ``max_pending`` jobs are already queued or running.
        """
        # Workers delete entries once done, so the queue length is exactly
        # the queued + running jobs. Check-then-add is not atomic, so two
        # racing submits can overshoot by one: fine for a soft limit.
        if await self._redis.xlen(QUEUE) >= self._max_pending:
            raise QueueFullError(f"{self._max_pending} jobs are already queued or running")
        job = Job(
            id=uuid4().hex,
            question=question,
            researcher=researcher,
            status=JobStatus.QUEUED,
            created_at=_now(),
        )
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.set(_job_key(job.id), job.model_dump_json())
            _add_event(pipe, job.id, "queued", "waiting for a worker")
            pipe.xadd(QUEUE, {"job_id": job.id})
            await pipe.execute()
        return job

    async def get(self, job_id: str) -> Job | None:
        """The job, or None if it does not exist (or expired)."""
        raw = await self._redis.get(_job_key(job_id))
        return None if raw is None else Job.model_validate_json(raw)

    async def events(
        self, job_id: str, after: str = "0-0", block_ms: int | None = None
    ) -> list[JobEvent]:
        """Events after the entry id ``after``, waiting up to ``block_ms`` for one if none."""
        try:
            reply = await self._redis.xread({_events_key(job_id): after}, count=100, block=block_ms)
        except RedisTimeoutError:
            if block_ms is None:
                raise
            return []  # an idle wait that ran long: same as no new events
        return [
            JobEvent.model_validate({"id": entry_id, **fields})
            for entry_id, fields in _entries(reply)
        ]

    async def cancel(self, job_id: str) -> Job:
        """Cancel a queued job now, or ask the worker to stop a running one.

        Raises:
            JobNotFoundError: If there is no such job.
            JobFinishedError: If it already finished.
        """

        async def change(pipe: Pipeline) -> Job:
            job = await _read(pipe, job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            if job.finished:
                raise JobFinishedError(job_id)
            _begin(pipe)
            if job.status is JobStatus.QUEUED:
                # Its queue entry stays; the worker skips it (see start).
                return self._finish(pipe, job, JobStatus.CANCELLED, "cancelled before it started")
            job = job.model_copy(update={"cancel_requested": True})
            pipe.set(_job_key(job_id), job.model_dump_json())
            return job

        job: Job = await self._redis.transaction(change, _job_key(job_id), value_from_callable=True)
        return job

    # ----- worker side ----------------------------------------------------

    async def ensure_group(self) -> None:
        """Create the queue and its consumer group if they do not exist yet."""
        try:
            # id="0": deliver entries added before the group existed, too.
            await self._redis.xgroup_create(QUEUE, GROUP, id="0", mkstream=True)
        except ResponseError as err:
            if "BUSYGROUP" not in str(err):
                raise

    async def next_job(
        self, consumer: str, block_ms: int, claim_idle_ms: int
    ) -> tuple[str, str] | None:
        """The next ``(entry_id, job_id)`` for ``consumer``, or None if none came in ``block_ms``.

        Entries a dead worker left pending for over ``claim_idle_ms`` come
        first, then new ones.
        """
        # XAUTOCLAIM replies [next-start-id, [entry, ...], [deleted-id, ...]].
        reply: Any = await self._redis.xautoclaim(
            QUEUE, GROUP, consumer, min_idle_time=claim_idle_ms, count=1
        )
        entries = [cast(Entry, tuple(entry)) for entry in reply[1]]
        if not entries:
            try:
                reply = await self._redis.xreadgroup(
                    GROUP, consumer, {QUEUE: ">"}, count=1, block=block_ms
                )
            except RedisTimeoutError:
                return None  # an idle wait that ran long: same as an empty queue
            entries = _entries(reply)
        if not entries:
            return None
        entry_id, fields = entries[0]
        return entry_id, fields["job_id"]

    async def start(self, job_id: str, max_attempts: int) -> Job | None:
        """Mark the job running; None if it must not run.

        A job must not run if it is gone, already finished, was cancelled
        while waiting, or has used ``max_attempts`` (it is then failed).
        """

        async def change(pipe: Pipeline) -> Job | None:
            job = await _read(pipe, job_id)
            if job is None or job.finished:
                return None
            _begin(pipe)
            if job.cancel_requested:
                self._finish(pipe, job, JobStatus.CANCELLED, "cancelled before it restarted")
                return None
            if job.attempts >= max_attempts:
                error = f"gave up after {job.attempts} attempt(s): the worker stopped mid-job"
                self._finish(pipe, job, JobStatus.FAILED, error, error=error)
                return None
            job = job.model_copy(
                update={
                    "status": JobStatus.RUNNING,
                    "started_at": _now(),
                    "attempts": job.attempts + 1,
                }
            )
            pipe.set(_job_key(job_id), job.model_dump_json())
            _add_event(pipe, job_id, "started", f"worker started (attempt {job.attempts})")
            return job

        job: Job | None = await self._redis.transaction(
            change, _job_key(job_id), value_from_callable=True
        )
        return job

    async def add_event(
        self, job_id: str, type_: EventType, message: str, node: str | None = None
    ) -> None:
        """Append one event to the job's stream."""
        async with self._redis.pipeline(transaction=False) as pipe:
            _add_event(pipe, job_id, type_, message, node)
            await pipe.execute()

    async def cancel_requested(self, job_id: str) -> bool:
        """True if the API asked to stop this job."""
        job = await self.get(job_id)
        return job is not None and job.cancel_requested

    async def finish(
        self,
        job_id: str,
        status: JobStatus,
        message: str,
        *,
        report: Report | None = None,
        error: str | None = None,
    ) -> Job | None:
        """Move a running job to its final ``status``; None if it was already final."""

        async def change(pipe: Pipeline) -> Job | None:
            job = await _read(pipe, job_id)
            if job is None or job.finished:
                return None
            _begin(pipe)
            return self._finish(pipe, job, status, message, report=report, error=error)

        job: Job | None = await self._redis.transaction(
            change, _job_key(job_id), value_from_callable=True
        )
        return job

    async def requeue(self, entry_id: str, job_id: str) -> None:
        """Hand a job back to the queue under a new entry (this worker is shutting down).

        Safe whatever the worker had got to: a finished (or gone) job is
        only acknowledged, a job not yet started keeps its state, and a
        running one goes back to queued without counting the attempt,
        since stopping was not the job's fault.
        """

        async def change(pipe: Pipeline) -> None:
            job = await _read(pipe, job_id)
            _begin(pipe)
            if job is not None and not job.finished:
                if job.status is JobStatus.RUNNING:
                    job = job.model_copy(
                        update={"status": JobStatus.QUEUED, "attempts": job.attempts - 1}
                    )
                    pipe.set(_job_key(job_id), job.model_dump_json())
                    _add_event(pipe, job_id, "requeued", "worker shut down; waiting for another")
                pipe.xadd(QUEUE, {"job_id": job_id})
            pipe.xack(QUEUE, GROUP, entry_id)
            pipe.xdel(QUEUE, entry_id)

        await self._redis.transaction(change, _job_key(job_id))

    async def ack(self, entry_id: str) -> None:
        """Remove a handled entry from the queue (``XACK``, then ``XDEL`` to keep it short)."""
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.xack(QUEUE, GROUP, entry_id)
            pipe.xdel(QUEUE, entry_id)
            await pipe.execute()

    def _finish(
        self,
        pipe: Pipeline,
        job: Job,
        status: JobStatus,
        message: str,
        *,
        report: Report | None = None,
        error: str | None = None,
    ) -> Job:
        """Queue the writes that end ``job`` on ``pipe`` (already in MULTI) and return it."""
        job = job.model_copy(
            update={"status": status, "finished_at": _now(), "report": report, "error": error}
        )
        pipe.set(_job_key(job.id), job.model_dump_json(), ex=self._result_ttl_s)
        _add_event(pipe, job.id, cast(EventType, status.value), message)
        pipe.expire(_events_key(job.id), self._result_ttl_s)
        return job
