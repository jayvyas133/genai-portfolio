"""Tests for the HTTP chat API with a fake chat service."""

import asyncio
from collections.abc import Iterator

import psycopg
import pytest
from aicore.errors import ProviderTimeoutError
from flask.testing import FlaskClient
from text_to_sql_agent.agent import PastTurn
from text_to_sql_agent.aio import LoopThread
from text_to_sql_agent.api import create_app
from text_to_sql_agent.executor import QueryResult
from text_to_sql_agent.service import ConversationBusyError, ConversationNotFoundError
from text_to_sql_agent.turn import AgentTurn

CID = "6f1c1f7e-8a43-4f53-9b0a-0d7c3a1e2b11"


class _FakeChat:
    """In-memory ``Chat``; ``fail`` makes ``send`` raise, ``slow`` makes it hang."""

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.turns: dict[str, list[PastTurn]] = {}
        self.fail: Exception | None = None
        self.slow = False
        self.healthy = {"chinook": True, "memory": True}

    async def create(self) -> str:
        self.ids.append(CID)
        self.turns[CID] = []
        return CID

    async def send(self, conversation_id: str, question: str) -> AgentTurn:
        if conversation_id not in self.turns:
            raise ConversationNotFoundError(conversation_id)
        if self.fail is not None:
            raise self.fail
        if self.slow:
            await asyncio.sleep(10)
        self.turns[conversation_id].append(
            PastTurn(question=question, sql="SELECT 1", preview="(1 of 1 rows)")
        )
        return AgentTurn(
            question=question,
            sql="SELECT 1 AS n",
            result=QueryResult(columns=["n"], rows=[[1]]),
            answer="There is 1.",
            attempts=2,
        )

    async def history(self, conversation_id: str) -> list[PastTurn]:
        if conversation_id not in self.turns:
            raise ConversationNotFoundError(conversation_id)
        return self.turns[conversation_id]

    async def health(self) -> dict[str, bool]:
        return self.healthy


@pytest.fixture
def loop() -> Iterator[LoopThread]:
    thread = LoopThread()
    yield thread
    thread.stop()


@pytest.fixture
def chat() -> _FakeChat:
    return _FakeChat()


@pytest.fixture
def client(chat: _FakeChat, loop: LoopThread) -> FlaskClient:
    return create_app(chat, loop=loop, timeout_s=0.5).test_client()


def test_create_conversation_returns_its_id_and_location(client: FlaskClient) -> None:
    response = client.post("/conversations")
    assert response.status_code == 201
    assert response.get_json() == {"id": CID}
    assert response.headers["Location"] == f"/conversations/{CID}"


def test_message_returns_the_answer_sql_and_rows(client: FlaskClient) -> None:
    client.post("/conversations")
    response = client.post(f"/conversations/{CID}/messages", json={"question": "  How many?  "})
    assert response.status_code == 200
    body = response.get_json()
    assert body["question"] == "How many?"  # whitespace stripped
    assert body["answer"] == "There is 1." and body["sql"] == "SELECT 1 AS n"
    assert body["columns"] == ["n"] and body["rows"] == [[1]]
    assert body["attempts"] == 2 and body["error_kind"] is None
    assert body["latency_ms"] >= 0


def test_history_lists_remembered_turns(client: FlaskClient) -> None:
    client.post("/conversations")
    assert client.get(f"/conversations/{CID}").get_json() == {"id": CID, "turns": []}
    client.post(f"/conversations/{CID}/messages", json={"question": "q1"})
    turns = client.get(f"/conversations/{CID}").get_json()["turns"]
    assert [t["question"] for t in turns] == ["q1"]


@pytest.mark.parametrize(
    ("body", "error"),
    [
        (None, "invalid_json"),
        ({"question": ""}, "invalid_request"),
        ({"question": "   "}, "invalid_request"),
        ({"question": "x" * 2001}, "invalid_request"),
        ({"question": "q", "extra": 1}, "invalid_request"),
    ],
)
def test_bad_bodies_are_400(client: FlaskClient, body: object, error: str) -> None:
    client.post("/conversations")
    response = client.post(f"/conversations/{CID}/messages", json=body)
    assert response.status_code == 400 and response.get_json()["error"] == error


def test_oversized_body_is_413(client: FlaskClient) -> None:
    response = client.post(
        f"/conversations/{CID}/messages",
        data="x" * (17 * 1024),
        content_type="application/json",
    )
    assert response.status_code == 413


def test_unknown_conversation_is_404(client: FlaskClient) -> None:
    assert client.get("/conversations/nope").status_code == 404
    response = client.post("/conversations/nope/messages", json={"question": "q"})
    assert response.status_code == 404 and response.get_json()["error"] == "not_found"


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (ConversationBusyError(CID), 409, "busy"),
        (ProviderTimeoutError("slow"), 502, "model_error"),
        (psycopg.OperationalError("down"), 503, "database_unavailable"),
    ],
)
def test_service_failures_map_to_status_codes(
    client: FlaskClient, chat: _FakeChat, error: Exception, status: int, code: str
) -> None:
    client.post("/conversations")
    chat.fail = error
    response = client.post(f"/conversations/{CID}/messages", json={"question": "q"})
    assert response.status_code == status and response.get_json()["error"] == code


def test_a_turn_over_the_time_limit_is_504(client: FlaskClient, chat: _FakeChat) -> None:
    client.post("/conversations")
    chat.slow = True
    response = client.post(f"/conversations/{CID}/messages", json={"question": "q"})
    assert response.status_code == 504 and response.get_json()["error"] == "timeout"


def test_health_is_503_when_a_database_is_down(client: FlaskClient, chat: _FakeChat) -> None:
    assert client.get("/healthz").status_code == 200
    chat.healthy = {"chinook": True, "memory": False}
    response = client.get("/healthz")
    assert response.status_code == 503 and response.get_json() == chat.healthy
