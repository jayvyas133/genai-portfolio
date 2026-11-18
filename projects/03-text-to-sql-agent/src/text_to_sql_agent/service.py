"""The chat service behind the HTTP API: conversations, turns and their memory.

One ``ChatService`` per worker process, created once on the API's event
loop thread (``aio.LoopThread``). It owns two connection pools, one per
role, and never mixes them:

- **data pool**, as ``sql_agent_ro`` on ``chinook``: the agent's SQL.
- **memory pool**, as ``sql_agent_app`` on ``text_to_sql_memory``: the
  LangGraph checkpoints, the conversation table and the turn locks.

Conversations survive restarts and are shared by every gunicorn worker,
because their state is in Postgres, not in process memory.
"""

import uuid
from typing import Any, Protocol

import psycopg
from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.providers.factory import build_client
from aicore.tracing import LogTracer
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from text_to_sql_agent.agent import ConversationalAgent, PastTurn
from text_to_sql_agent.config import TextToSqlSettings
from text_to_sql_agent.executor import SqlExecutor
from text_to_sql_agent.schema import load_schema
from text_to_sql_agent.turn import AgentTurn

# First key of the two-key advisory lock: a namespace, so these locks
# cannot collide with any other advisory lock in the same database.
_LOCK_NAMESPACE = 7319


class ConversationNotFoundError(LookupError):
    """No conversation has this id."""


class ConversationBusyError(RuntimeError):
    """Another request is answering a question in this conversation right now."""


class Chat(Protocol):
    """What the API needs from the service (a fake in API tests)."""

    async def create(self) -> str:
        """Start a conversation and return its id."""
        ...

    async def send(self, conversation_id: str, question: str) -> AgentTurn:
        """Answer the next question in a conversation."""
        ...

    async def history(self, conversation_id: str) -> list[PastTurn]:
        """The remembered turns of a conversation."""
        ...

    async def health(self) -> dict[str, bool]:
        """Whether each database answers."""
        ...


class ChatService:
    """``Chat`` backed by Postgres: the agent plus durable conversation memory."""

    def __init__(
        self,
        agent: ConversationalAgent,
        client: LLMClient,
        data_pool: AsyncConnectionPool,
        memory_pool: "AsyncConnectionPool[psycopg.AsyncConnection[DictRow]]",
    ) -> None:
        """Use ``agent``; ``client`` and both pools are closed by ``aclose``."""
        self._agent = agent
        self._client = client
        self._data = data_pool
        self._memory = memory_pool

    @classmethod
    async def open(cls, settings: TextToSqlSettings, llm: LLMSettings) -> "ChatService":
        """Open both pools, read the schema and build the agent.

        Fails fast if a database is unreachable, so a misconfigured worker
        does not start and then answer every request with an error.

        Raises:
            psycopg_pool.PoolTimeout: If a database cannot be reached.
            psycopg.Error: If the schema cannot be read.
        """
        size = settings.pool_max_size
        data = AsyncConnectionPool(settings.database_url, min_size=1, max_size=size, open=False)
        # Each turn holds one memory connection for its lock while the
        # checkpointer needs another, so this pool is twice as large:
        # with only `size` connections, `size` concurrent turns would hold
        # them all and wait forever for the checkpointer.
        memory = AsyncConnectionPool(
            settings.memory_database_url,
            connection_class=psycopg.AsyncConnection[DictRow],
            min_size=1,
            max_size=2 * size,
            open=False,
            # What AsyncPostgresSaver expects of its connections.
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        await data.open(wait=True, timeout=10)
        await memory.open(wait=True, timeout=10)
        schema = await load_schema(settings.database_url)
        client = build_client(llm, tracer=LogTracer())
        executor = SqlExecutor(
            settings.database_url, settings.statement_timeout_ms, settings.max_rows, pool=data
        )
        agent = ConversationalAgent(
            client, executor, schema, settings.agent_limits(), AsyncPostgresSaver(memory)
        )
        return cls(agent, client, data, memory)

    async def aclose(self) -> None:
        """Close the model client and both pools."""
        await self._client.aclose()
        await self._data.close()
        await self._memory.close()

    async def create(self) -> str:
        """Start a conversation; the id is server-generated, so it cannot be guessed or reused."""
        conversation_id = str(uuid.uuid4())
        async with self._memory.connection() as conn:
            await conn.execute("INSERT INTO conversation (id) VALUES (%s)", (conversation_id,))
        return conversation_id

    async def send(self, conversation_id: str, question: str) -> AgentTurn:
        """Answer ``question`` as the next turn, one turn at a time per conversation.

        A Postgres advisory lock (not a process lock) makes "one turn at a
        time" hold across every worker process. Two concurrent turns would
        both start from the same checkpoint and one would overwrite the
        other's history.

        Raises:
            ConversationNotFoundError: If the id was never issued.
            ConversationBusyError: If a turn is already running for it.
            LLMError: If a model call fails after retries.
        """
        await self._require(conversation_id)
        key = (_LOCK_NAMESPACE, conversation_id)
        async with self._memory.connection() as conn:
            row = await (
                await conn.execute("SELECT pg_try_advisory_lock(%s, hashtext(%s)) AS ok", key)
            ).fetchone()
            if not _first(row):
                raise ConversationBusyError(conversation_id)
            try:
                return await self._agent.ask(conversation_id, question)
            finally:
                # Session-level lock on a pooled connection: release it
                # explicitly, or it would stay held after the connection
                # goes back to the pool.
                await conn.execute("SELECT pg_advisory_unlock(%s, hashtext(%s))", key)

    async def history(self, conversation_id: str) -> list[PastTurn]:
        """The remembered turns, oldest first; empty for a new conversation.

        Raises:
            ConversationNotFoundError: If the id was never issued.
        """
        await self._require(conversation_id)
        return await self._agent.history(conversation_id)

    async def health(self) -> dict[str, bool]:
        """Whether each pool can run ``SELECT 1``."""
        return {"chinook": await _ping(self._data), "memory": await _ping(self._memory)}

    async def _require(self, conversation_id: str) -> None:
        """Raise ``ConversationNotFoundError`` unless the id was issued by ``create``."""
        try:
            uuid.UUID(conversation_id)
        except ValueError as err:
            raise ConversationNotFoundError(conversation_id) from err
        async with self._memory.connection() as conn:
            row = await (
                await conn.execute("SELECT 1 FROM conversation WHERE id = %s", (conversation_id,))
            ).fetchone()
        if row is None:
            raise ConversationNotFoundError(conversation_id)


def _first(row: Any) -> Any:
    """The first value of a row from either a tuple or a dict row factory."""
    if row is None:
        return None
    return next(iter(row.values())) if isinstance(row, dict) else row[0]


async def _ping(pool: "AsyncConnectionPool[Any]") -> bool:
    """Whether ``pool`` hands out a connection that answers ``SELECT 1`` within 2 s."""
    try:
        async with pool.connection(timeout=2) as conn:
            await conn.execute("SELECT 1")
            await conn.rollback()
    except psycopg.Error:  # includes psycopg_pool.PoolTimeout
        return False
    return True
