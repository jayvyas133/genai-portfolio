"""HTTP API for Deep Research: submit research jobs, follow them live, fetch the reports.

| Method and path                 | What it does                                          |
|---------------------------------|-------------------------------------------------------|
| ``POST /research``              | queue a job; ``202 Accepted`` + ``Location`` header   |
| ``GET /research/{id}``          | the job: status, and the report once it is done       |
| ``GET /research/{id}/events``   | progress as Server-Sent Events (SSE), live            |
| ``DELETE /research/{id}``       | cancel it                                             |
| ``GET /healthz``                | liveness, including the Redis connection              |

The API never runs research itself; it only reads and writes Redis
(``jobs.py``), so it holds no state and any number of replicas can run.
The worker (``worker.py``) does the research.

Run it with ``uv run uvicorn --factory deep_research.api:create_app``.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, cast

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from redis.asyncio import Redis

from deep_research.config import DeepResearchSettings
from deep_research.jobs import (
    MAX_BLOCK_S,
    Job,
    JobEvent,
    JobFinishedError,
    JobNotFoundError,
    JobStore,
    QueueFullError,
    connect,
    is_event_id,
)
from deep_research.report import ResearcherName

HEARTBEAT_S = MAX_BLOCK_S
"""Idle seconds before the event stream sends a comment line, so proxies keep it open."""

RETRY_AFTER_S = 30
"""Suggested wait sent with 429 when the queue is full."""


class ResearchRequest(BaseModel):
    """Body of ``POST /research``."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=500)
    researcher: ResearcherName = "agents"


def get_store(request: Request) -> JobStore:
    """The app's ``JobStore`` (created in the lifespan)."""
    return cast(JobStore, request.app.state.store)


Store = Annotated[JobStore, Depends(get_store)]

router = APIRouter()


@router.post("/research", status_code=202, response_model=Job)
async def submit(body: ResearchRequest, store: Store, response: Response) -> Job:
    """Queue a research job; poll the ``Location`` URL or stream ``.../events``."""
    try:
        job = await store.submit(body.question.strip(), body.researcher)
    except QueueFullError as err:
        raise HTTPException(
            429, detail=str(err), headers={"Retry-After": str(RETRY_AFTER_S)}
        ) from err
    response.headers["Location"] = f"/research/{job.id}"
    return job


@router.get("/research/{job_id}", response_model=Job)
async def get_job(job_id: str, store: Store) -> Job:
    """The job's status, and its report once done."""
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(404, detail="job not found (or expired)")
    return job


@router.delete("/research/{job_id}", status_code=202, response_model=Job)
async def cancel_job(job_id: str, store: Store) -> Job:
    """Cancel the job: at once if queued, within about a second if running."""
    try:
        return await store.cancel(job_id)
    except JobNotFoundError as err:
        raise HTTPException(404, detail="job not found (or expired)") from err
    except JobFinishedError as err:
        raise HTTPException(409, detail="job already finished") from err


@router.get("/research/{job_id}/events")
async def stream_events(
    job_id: str,
    store: Store,
    last_event_id: Annotated[str | None, Header()] = None,
) -> StreamingResponse:
    """Stream the job's events as SSE, from the start or after ``Last-Event-ID``.

    Every past event is replayed first, so a client can connect at any
    time; the stream ends after the final event (done, failed or cancelled).
    """
    if last_event_id is not None and not is_event_id(last_event_id):
        raise HTTPException(400, detail="Last-Event-ID is not an event id from this stream")
    if await store.get(job_id) is None:
        raise HTTPException(404, detail="job not found (or expired)")
    return StreamingResponse(
        _sse(store, job_id, last_event_id or "0-0"),
        media_type="text/event-stream",
        # no-cache: never serve a stale stream; X-Accel-Buffering: stop
        # nginx-style proxies from holding events back in a buffer.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/healthz")
async def healthz(store: Store) -> dict[str, str]:
    """``{"status": "ok"}`` if Redis answers, else 503."""
    try:
        await store.ping()
    except Exception as err:  # any Redis failure means "not healthy"
        raise HTTPException(503, detail="redis unavailable") from err
    return {"status": "ok"}


async def _sse(store: JobStore, job_id: str, after: str) -> AsyncIterator[str]:
    """SSE frames for every event after ``after``, until the final one."""
    block_ms = int(HEARTBEAT_S * 1000)
    while True:
        events = await store.events(job_id, after, block_ms=block_ms)
        if not events:
            if await store.get(job_id) is None:  # expired while we waited
                return
            yield ": keep-alive\n\n"
            continue
        for event in events:
            yield format_sse(event)
            after = event.id
            if event.final:
                return


def format_sse(event: JobEvent) -> str:
    """One SSE frame: ``id`` (for Last-Event-ID), ``event`` (the type), ``data`` (JSON)."""
    return f"id: {event.id}\nevent: {event.type}\ndata: {event.model_dump_json()}\n\n"


def create_app(settings: DeepResearchSettings | None = None, redis: Redis | None = None) -> FastAPI:
    """Build the app; ``redis`` lets tests pass an in-memory fake."""
    settings = settings or DeepResearchSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client = redis or connect(settings.redis_url)
        app.state.store = JobStore(
            client, max_pending=settings.max_pending_jobs, result_ttl_s=settings.result_ttl_s
        )
        try:
            yield
        finally:
            if redis is None:
                await client.aclose()

    app = FastAPI(title="Deep Research", version="0.1.0", lifespan=lifespan)
    app.include_router(router)
    return app
