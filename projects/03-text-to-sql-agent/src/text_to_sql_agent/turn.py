"""What one chat turn produces, and the interface every agent implements.

The eval and (in M3) the HTTP API drive an agent the way a chat client
does: a conversation id plus the next question. Whether the agent
remembers earlier turns is up to the agent; the single-shot baseline
does not, which is exactly what the eval's follow-up questions expose.
"""

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from text_to_sql_agent.executor import QueryResult

ErrorKind = Literal["no_sql", "unsafe", "execution"]
"""Why a turn has no result: the reply held no SQL, the guard rejected
it, or Postgres rejected it. Model-call failures are raised, not returned."""


class AgentTurn(BaseModel):
    """The agent's answer to one question: the SQL it ran and the rows, or why not."""

    model_config = ConfigDict(frozen=True)

    question: str
    sql: str | None = None
    result: QueryResult | None = None
    error_kind: ErrorKind | None = None
    error: str | None = None


class Agent(Protocol):
    """Anything that answers the next question in a conversation."""

    @property
    def name(self) -> str:
        """Short id saved with eval results, e.g. ``single-shot``."""
        ...

    async def ask(self, thread_id: str, question: str) -> AgentTurn:
        """Answer ``question`` as the next turn of conversation ``thread_id``.

        Raises:
            LLMError: If a model call fails after retries.
        """
        ...
