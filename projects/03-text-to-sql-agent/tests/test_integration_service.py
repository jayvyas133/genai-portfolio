"""Integration tests for the chat service on real Postgres (memory + Chinook).

Run ``uv run text-to-sql load-db`` and ``uv run text-to-sql init-memory``
first. The model is aicore's offline fake provider, so no model runs:
these tests are about memory, locks and roles, not answer quality.
"""

from collections.abc import AsyncIterator

import psycopg
import pytest
from aicore.config import LLMSettings
from text_to_sql_agent.config import TextToSqlSettings
from text_to_sql_agent.service import (
    ChatService,
    ConversationBusyError,
    ConversationNotFoundError,
)

pytestmark = pytest.mark.integration

SETTINGS = TextToSqlSettings(_env_file=None)
FAKE_LLM = LLMSettings(_env_file=None, provider="fake", model="fake-model")


@pytest.fixture
async def service() -> AsyncIterator[ChatService]:
    svc = await ChatService.open(SETTINGS, FAKE_LLM)
    yield svc
    await svc.aclose()


async def test_memory_survives_a_new_service_instance(service: ChatService) -> None:
    cid = await service.create()
    assert await service.history(cid) == []
    turn = await service.send(cid, "How many tracks?")
    # The fake model writes no SQL: the turn fails after its attempts,
    # but it is still remembered, with its error.
    assert turn.error_kind == "no_sql" and turn.attempts == SETTINGS.max_attempts

    fresh = await ChatService.open(SETTINGS, FAKE_LLM)  # as after a restart
    try:
        [past] = await fresh.history(cid)
    finally:
        await fresh.aclose()
    assert past["question"] == "How many tracks?"
    assert past["preview"].startswith("failed:")


async def test_unknown_and_malformed_ids_are_not_found(service: ChatService) -> None:
    for cid in ("not-a-uuid", "00000000-0000-4000-8000-000000000000"):
        with pytest.raises(ConversationNotFoundError):
            await service.history(cid)
        with pytest.raises(ConversationNotFoundError):
            await service.send(cid, "q")


async def test_a_second_turn_in_the_same_conversation_is_refused_while_one_runs(
    service: ChatService,
) -> None:
    cid = await service.create()
    # Hold the conversation's lock from another session, as a turn in
    # another worker process would.
    async with await psycopg.AsyncConnection.connect(
        SETTINGS.memory_database_url, autocommit=True
    ) as other:
        await other.execute("SELECT pg_advisory_lock(7319, hashtext(%s))", (cid,))
        with pytest.raises(ConversationBusyError):
            await service.send(cid, "q")
        await other.execute("SELECT pg_advisory_unlock(7319, hashtext(%s))", (cid,))
    turn = await service.send(cid, "q")  # free again
    assert turn.question == "q"


async def test_both_databases_report_healthy(service: ChatService) -> None:
    assert await service.health() == {"chinook": True, "memory": True}


async def test_the_memory_role_cannot_reach_chinook() -> None:
    url = SETTINGS.memory_database_url.rsplit("/", 1)[0] + "/chinook"
    with pytest.raises(psycopg.OperationalError, match="permission denied"):
        await psycopg.AsyncConnection.connect(url)
