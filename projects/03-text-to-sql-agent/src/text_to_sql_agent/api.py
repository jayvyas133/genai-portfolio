"""HTTP chat API over the conversational agent.

    POST /conversations                 -> 201 {"id": ...}
    POST /conversations/<id>/messages   -> 200 answer, SQL and rows
    GET  /conversations/<id>            -> 200 remembered turns
    GET  /healthz                       -> 200, or 503 if a database is down

Run it in the container with gunicorn: ``gunicorn "text_to_sql_agent.api:create_app()"``.

Views are synchronous (Flask) and the service is async. Each worker
process starts one ``aio.LoopThread``; the service and its connection
pools live on that loop, and views hand coroutines to it. Pools and the
model client are therefore shared by all requests of a worker instead of
being opened per request (ADR 0020).
"""

import atexit
import logging
import time
from typing import Annotated

import psycopg
from aicore.config import LLMSettings
from aicore.errors import LLMError
from flask import Flask, Response, jsonify, request
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from text_to_sql_agent.agent import PastTurn
from text_to_sql_agent.aio import LoopThread
from text_to_sql_agent.config import TextToSqlSettings
from text_to_sql_agent.executor import Value
from text_to_sql_agent.service import (
    Chat,
    ChatService,
    ConversationBusyError,
    ConversationNotFoundError,
)
from text_to_sql_agent.turn import ErrorKind

logger = logging.getLogger(__name__)

# A question is at most 2,000 characters; 16 KB leaves room for JSON
# escaping and rejects huge bodies before they are parsed.
MAX_BODY_BYTES = 16 * 1024
# Health checks and conversation lookups are quick; only turns get the
# long request timeout.
_SHORT_TIMEOUT_S = 10.0


class MessageRequest(BaseModel):
    """Body of ``POST /conversations/<id>/messages``."""

    model_config = ConfigDict(extra="forbid")

    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]


class MessageResponse(BaseModel):
    """One answered turn: the reply a user reads, and the SQL and rows behind it.

    ``error_kind`` and ``error`` are set when no query succeeded; the
    turn still returns 200, because the conversation itself worked.
    """

    conversation_id: str
    question: str
    answer: str | None
    sql: str | None
    columns: list[str]
    rows: list[list[Value]]
    truncated: bool
    attempts: int
    error_kind: ErrorKind | None
    error: str | None
    latency_ms: float


class ConversationResponse(BaseModel):
    """Body of ``GET /conversations/<id>``: the turns the agent remembers."""

    id: str
    turns: list[PastTurn]


def _error(status: int, error: str, detail: str) -> tuple[Response, int]:
    """A JSON error body: a stable ``error`` code and a human ``detail``."""
    return jsonify(error=error, detail=detail), status


def _validation_detail(err: ValidationError) -> str:
    """Summarise ``err`` as ``field: message`` pairs, without echoing the input."""
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or 'body'}: {e['msg']}" for e in err.errors()
    )


def create_app(
    chat: Chat | None = None, *, loop: LoopThread | None = None, timeout_s: float | None = None
) -> Flask:
    """Build the Flask app.

    Args:
        chat: The service. Defaults to ``ChatService`` on Postgres and the
            model configured by ``AICORE_*``; tests pass a fake.
        loop: The event-loop thread the service runs on; one is started
            when omitted.
        timeout_s: Overrides ``TEXT_TO_SQL_REQUEST_TIMEOUT_S`` for turns.
    """
    settings = TextToSqlSettings()
    limit_s = timeout_s if timeout_s is not None else settings.request_timeout_s
    loop = loop or LoopThread()
    if chat is None:
        service = loop.run(ChatService.open(settings, LLMSettings()), timeout_s=60)
        atexit.register(lambda: loop.run(service.aclose(), timeout_s=10))
        chat = service
    backend: Chat = chat

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES
    app.extensions["aio_loop"] = loop

    @app.errorhandler(413)
    def too_large(_: Exception) -> tuple[Response, int]:
        return _error(413, "too_large", f"The body is over {MAX_BODY_BYTES // 1024} KB.")

    @app.errorhandler(ConversationNotFoundError)
    def not_found(_: Exception) -> tuple[Response, int]:
        return _error(404, "not_found", "No conversation has this id.")

    @app.errorhandler(ConversationBusyError)
    def busy(_: Exception) -> tuple[Response, int]:
        return _error(
            409, "busy", "This conversation is answering another question; retry when it is done."
        )

    @app.errorhandler(TimeoutError)
    def timed_out(_: Exception) -> tuple[Response, int]:
        logger.warning("Request timed out")
        return _error(504, "timeout", "No answer in time; try again.")

    @app.errorhandler(LLMError)
    def model_failed(_: Exception) -> tuple[Response, int]:
        logger.exception("Model provider call failed")
        return _error(502, "model_error", "The model provider failed; try again.")

    @app.errorhandler(psycopg.OperationalError)
    def database_down(_: Exception) -> tuple[Response, int]:
        logger.exception("Postgres unreachable")
        return _error(503, "database_unavailable", "A database is unavailable.")

    @app.get("/healthz")
    def health() -> tuple[Response, int]:
        status = loop.run(backend.health(), _SHORT_TIMEOUT_S)
        return jsonify(status), 200 if all(status.values()) else 503

    @app.post("/conversations")
    def create_conversation() -> tuple[Response, int]:
        conversation_id = loop.run(backend.create(), _SHORT_TIMEOUT_S)
        response = jsonify(id=conversation_id)
        response.headers["Location"] = f"/conversations/{conversation_id}"
        return response, 201

    @app.get("/conversations/<conversation_id>")
    def get_conversation(conversation_id: str) -> tuple[Response, int]:
        turns = loop.run(backend.history(conversation_id), _SHORT_TIMEOUT_S)
        body = ConversationResponse(id=conversation_id, turns=turns)
        return jsonify(body.model_dump(mode="json")), 200

    @app.post("/conversations/<conversation_id>/messages")
    def send_message(conversation_id: str) -> tuple[Response, int]:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _error(400, "invalid_json", 'Send a JSON object: {"question": "..."}.')
        try:
            body = MessageRequest.model_validate(payload)
        except ValidationError as err:
            return _error(400, "invalid_request", _validation_detail(err))
        start = time.perf_counter()
        turn = loop.run(backend.send(conversation_id, body.question), limit_s)
        result = turn.result
        response = MessageResponse(
            conversation_id=conversation_id,
            question=body.question,
            answer=turn.answer,
            sql=turn.sql,
            columns=result.columns if result else [],
            rows=result.rows if result else [],
            truncated=result.truncated if result else False,
            attempts=turn.attempts,
            error_kind=turn.error_kind,
            error=turn.error,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )
        return jsonify(response.model_dump(mode="json")), 200

    return app
