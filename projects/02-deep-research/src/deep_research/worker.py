"""The research worker: takes jobs from the Redis queue and runs them.

Run it with ``uv run deep-research worker``; run more processes (or
containers) to work through the queue faster. Each process runs
``concurrency`` loops, each one a separate consumer in the ``workers``
group, so Redis hands every job to exactly one of them.

For one job, a loop:

1. marks it running (``JobStore.start``; skipped if it was cancelled or
   has no attempts left),
2. runs the researcher in a task and, while it runs, polls once a
   second for a cancel request, under a hard ``job_timeout_s`` limit,
3. writes the final state (done with the report, failed with the error,
   or cancelled) and acknowledges the queue entry.

Every finished graph step is published to the job's event stream as it
happens, which is what the API streams to clients.

Shutdown (SIGTERM from Docker, or Ctrl+C) stops the running research
and puts the job straight back in the queue for another worker. A worker
that dies without shutting down (killed, out of memory) cannot do that:
its job stays pending in Redis and is claimed by another worker once it
has been idle for ``job_timeout_s`` plus a grace period. A job cannot
legitimately run longer than the timeout, so a live job is never claimed
twice.
"""

import asyncio
import contextlib
import logging
import os
import signal
import socket
from collections.abc import Callable, Coroutine
from typing import Any

from deep_research.agents import OnEvent, ProgressEvent
from deep_research.jobs import Job, JobStatus, JobStore
from deep_research.report import Report, ResearcherName

ResearchFn = Callable[[str, ResearcherName, OnEvent], Coroutine[Any, Any, Report]]
"""Runs one question with the named researcher, publishing steps via the callback."""

MAX_ATTEMPTS = 2
"""Starts per job; the second covers one worker crash, a third would likely crash again."""

CLAIM_GRACE_S = 60.0
"""Added to the job timeout before a dead worker's job is claimed by another."""

logger = logging.getLogger("deep_research.worker")


class Worker:
    """Pulls jobs from a ``JobStore`` and runs them with ``research``."""

    def __init__(
        self,
        store: JobStore,
        research: ResearchFn,
        *,
        concurrency: int = 1,
        job_timeout_s: float = 900.0,
        name: str | None = None,
        block_ms: int = 5_000,
        cancel_poll_s: float = 1.0,
    ) -> None:
        """Configure the worker; ``name`` defaults to ``<host>-<pid>``."""
        self._store = store
        self._research = research
        self._concurrency = concurrency
        self._job_timeout_s = job_timeout_s
        self.name = name or f"{socket.gethostname()}-{os.getpid()}"
        self.block_ms = block_ms
        self._cancel_poll_s = cancel_poll_s
        self._claim_idle_ms = int((job_timeout_s + CLAIM_GRACE_S) * 1000)

    async def run(self) -> None:
        """Process jobs until cancelled (SIGTERM or Ctrl+C)."""
        await self._store.ensure_group()
        main = asyncio.current_task()
        if main is not None:
            # Not on Windows; there Ctrl+C still cancels via asyncio.run.
            with contextlib.suppress(NotImplementedError):
                asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, main.cancel)
        logger.info("worker %s started, %d slot(s)", self.name, self._concurrency)
        async with asyncio.TaskGroup() as group:
            for slot in range(self._concurrency):
                group.create_task(self._loop(f"{self.name}-{slot}"))

    async def _loop(self, consumer: str) -> None:
        """One slot: take a job, run it, repeat."""
        while True:
            claimed = await self._store.next_job(consumer, self.block_ms, self._claim_idle_ms)
            if claimed is not None:
                await self.process(*claimed)

    async def process(self, entry_id: str, job_id: str) -> None:
        """Run one queued job to a final state, then acknowledge its queue entry."""
        # Shutdown can strike at any await here, including while ``start``
        # or the final ack runs, so all of it is covered; ``requeue`` works
        # out from the job's state whether anything still needs handing back.
        try:
            job = await self._store.start(job_id, MAX_ATTEMPTS)
            if job is not None:
                logger.info("job %s started (attempt %d)", job.id, job.attempts)
                await self._execute(job)
            await self._store.ack(entry_id)
        except asyncio.CancelledError:
            # shield() so the requeue is not cut short by the same cancellation.
            logger.info("shutting down, handing job %s back to the queue", job_id)
            await asyncio.shield(self._store.requeue(entry_id, job_id))
            raise

    async def _execute(self, job: Job) -> None:
        """Run the research under the timeout, stop it on a cancel request, record the result."""

        async def publish(event: ProgressEvent) -> None:
            await self._store.add_event(job.id, "progress", event.message, node=event.node)

        task = asyncio.create_task(self._research(job.question, job.researcher, publish))
        cancelled = timed_out = False
        try:
            async with asyncio.timeout(self._job_timeout_s):
                cancelled = await self._wait_unless_cancelled(task, job.id)
        except TimeoutError:
            timed_out = True
        finally:
            # Stop the research before writing a final state, so it can
            # never publish an event after the final one.
            if not task.done():
                task.cancel()
                await asyncio.wait({task})

        if cancelled:
            await self._store.finish(job.id, JobStatus.CANCELLED, "cancelled by request")
        elif timed_out:
            error = f"timed out after {self._job_timeout_s:.0f} s"
            await self._store.finish(job.id, JobStatus.FAILED, error, error=error)
        elif (err := task.exception()) is not None:
            # A worker must outlive any one job, so every error ends the job, not the worker.
            logger.error("job %s failed", job.id, exc_info=err)
            error = f"{type(err).__name__}: {err}"
            await self._store.finish(job.id, JobStatus.FAILED, error, error=error)
        else:
            report = task.result()
            message = f"report ready, {len(report.sources)} source(s)"
            await self._store.finish(job.id, JobStatus.DONE, message, report=report)
        logger.info("job %s finished", job.id)

    async def _wait_unless_cancelled(self, task: asyncio.Task[Report], job_id: str) -> bool:
        """Wait for ``task``; True if a cancel request came first."""
        while True:
            done, _ = await asyncio.wait({task}, timeout=self._cancel_poll_s)
            if done:
                return False
            if await self._store.cancel_requested(job_id):
                return True
