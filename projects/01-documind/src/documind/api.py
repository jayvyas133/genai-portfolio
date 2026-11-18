"""HTTP API: ``POST /ask`` answers a question with cited sources; ``GET /health``.

Run it locally with ``uv run documind serve`` (Flask's development
server), or in the container with gunicorn:
``gunicorn "documind.api:create_app()"``.

Views are synchronous and each request runs the async pipeline on its
own event loop (``aio.run``). Async objects are bound to the loop that
created them, so the database connection and the model client are
opened and closed per request instead of shared (see ADR 0013).
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass

import psycopg
from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.errors import LLMError
from aicore.providers.factory import build_client
from aicore.tracing import InMemoryTracer
from flask import Flask, Response, jsonify, request
from pydantic import BaseModel, ConfigDict, ValidationError

from documind import aio
from documind.ask import answer_question, refusal_message
from documind.config import DocumindSettings
from documind.generation import Citation, RefusalReason
from documind.guards import answer_guards
from documind.retrieval import Retriever, RetrieverName, build_retriever
from documind.store import PgChunkStore, connect

logger = logging.getLogger(__name__)

# Chosen on the retrieval evals in M4 (ADR 0011).
RETRIEVER: RetrieverName = "hybrid-rerank"
# A question is at most 1,000 characters; 64 KB leaves room for JSON
# escaping and rejects huge bodies before they are parsed.
MAX_BODY_BYTES = 64 * 1024


@dataclass(frozen=True)
class Session:
    """What one request needs to answer a question."""

    retriever: Retriever
    client: LLMClient
    tracer: InMemoryTracer


SessionFactory = Callable[[], AbstractAsyncContextManager[Session]]


def database_sessions(settings: DocumindSettings, llm_settings: LLMSettings) -> SessionFactory:
    """Return a factory that opens a Postgres connection and a model client per request."""

    @asynccontextmanager
    async def open_session() -> AsyncIterator[Session]:
        tracer = InMemoryTracer()
        client = build_client(llm_settings, tracer=tracer, output_guards=answer_guards())
        try:
            async with await connect(settings.database_url) as conn:
                retriever = build_retriever(
                    RETRIEVER, PgChunkStore(conn), client, settings.embedding_dimensions
                )
                yield Session(retriever=retriever, client=client, tracer=tracer)
        finally:
            await client.aclose()

    return open_session


class AskRequest(BaseModel):
    """Body of ``POST /ask``."""

    model_config = ConfigDict(extra="forbid")

    question: str


class AskResponse(BaseModel):
    """Answer to ``POST /ask``.

    ``message`` explains a refusal and is ``None`` when answered.
    ``latency_ms`` and ``cost_usd`` cover the whole request (every
    embed, rerank and model call).
    """

    answer: str
    refused: bool
    refusal_reason: RefusalReason | None
    message: str | None
    citations: list[Citation]
    latency_ms: float
    cost_usd: float


async def answer_within(
    question: str, open_session: SessionFactory, timeout_s: float
) -> AskResponse:
    """Answer ``question`` in a fresh session, giving up after ``timeout_s``.

    Raises:
        TimeoutError: If the whole request took longer than ``timeout_s``.
        LLMError: If a model call failed after retries.
        psycopg.OperationalError: If Postgres is unreachable.
    """
    start = time.perf_counter()
    async with asyncio.timeout(timeout_s), open_session() as session:
        answer = await answer_question(question, session.retriever, session.client)
    return AskResponse(
        answer=answer.text,
        refused=answer.refused,
        refusal_reason=answer.refusal_reason,
        message=refusal_message(answer) if answer.refused else None,
        citations=answer.citations,
        latency_ms=round((time.perf_counter() - start) * 1000, 1),
        cost_usd=sum(c.cost_usd or 0.0 for c in session.tracer.records),
    )


def _error(status: int, error: str, detail: str) -> tuple[Response, int]:
    """A JSON error body: a stable ``error`` code and a human ``detail``."""
    return jsonify(error=error, detail=detail), status


def _validation_detail(err: ValidationError) -> str:
    """Summarise ``err`` as ``field: message`` pairs, without echoing the input."""
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or 'body'}: {e['msg']}" for e in err.errors()
    )


def create_app(
    open_session: SessionFactory | None = None, *, timeout_s: float | None = None
) -> Flask:
    """Build the Flask app.

    Args:
        open_session: Opens the per-request retriever and model client.
            Defaults to Postgres plus the model configured by ``AICORE_*``;
            tests pass a fake.
        timeout_s: Overrides ``DOCUMIND_REQUEST_TIMEOUT_S``.
    """
    settings = DocumindSettings()
    sessions = open_session or database_sessions(settings, LLMSettings())
    limit_s = timeout_s if timeout_s is not None else settings.request_timeout_s

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES

    @app.errorhandler(413)
    def too_large(_: Exception) -> tuple[Response, int]:
        return _error(413, "too_large", f"The body is over {MAX_BODY_BYTES // 1024} KB.")

    @app.get("/health")
    def health() -> Response:
        return jsonify(status="ok")

    @app.post("/ask")
    def ask() -> tuple[Response, int]:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _error(400, "invalid_json", 'Send a JSON object: {"question": "..."}.')
        try:
            body = AskRequest.model_validate(payload)
        except ValidationError as err:
            return _error(400, "invalid_request", _validation_detail(err))
        try:
            result = aio.run(answer_within(body.question, sessions, limit_s))
        except TimeoutError:
            logger.warning("Request timed out after %.0f s", limit_s)
            return _error(504, "timeout", f"No answer within {limit_s:.0f} s; try again.")
        except LLMError:
            logger.exception("Model provider call failed")
            return _error(502, "model_error", "The model provider failed; try again.")
        except psycopg.OperationalError:
            logger.exception("Postgres unreachable")
            return _error(503, "database_unavailable", "The document store is unavailable.")
        return jsonify(result.model_dump(mode="json")), 200

    return app
