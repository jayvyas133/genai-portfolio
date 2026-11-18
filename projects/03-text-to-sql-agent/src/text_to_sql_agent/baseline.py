"""Single-shot baseline: the whole schema in the prompt, one LLM call, run the SQL.

It has no memory and no retries. It is the cheapest Text-to-SQL system
that could work, and the bar the M2 agent has to clear: on single
questions it shows what one call can do, and on follow-ups ("now only
2023") it shows what is lost without conversation memory.
"""

from aicore.client import LLMClient

from text_to_sql_agent.executor import QueryRunner, SqlExecutionError
from text_to_sql_agent.generation import build_sql_request, extract_sql
from text_to_sql_agent.guard import UnsafeSqlError
from text_to_sql_agent.schema import Schema
from text_to_sql_agent.turn import AgentTurn


class SingleShotAgent:
    """``Agent`` that answers every question on its own, ignoring the conversation."""

    name = "single-shot"

    def __init__(self, client: LLMClient, executor: QueryRunner, schema: Schema) -> None:
        """Write SQL with ``client`` against ``schema`` and run it with ``executor``."""
        self._client = client
        self._executor = executor
        self._ddl = schema.ddl()

    async def ask(self, thread_id: str, question: str) -> AgentTurn:
        """Answer ``question``; ``thread_id`` is ignored (no memory).

        Raises:
            LLMError: If the model call fails after retries.
        """
        response = await self._client.complete(build_sql_request(self._ddl, question))
        sql = extract_sql(response.text)
        if sql is None:
            return AgentTurn(
                question=question, error_kind="no_sql", error="The reply contained no SQL."
            )
        try:
            result = await self._executor.run(sql)
        except UnsafeSqlError as err:
            return AgentTurn(question=question, sql=sql, error_kind="unsafe", error=str(err))
        except SqlExecutionError as err:
            return AgentTurn(question=question, sql=sql, error_kind="execution", error=str(err))
        return AgentTurn(question=question, sql=sql, result=result)
