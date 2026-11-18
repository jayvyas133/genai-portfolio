"""Configuration for the Text-to-SQL agent.

Values come from environment variables prefixed with ``TEXT_TO_SQL_``
(or a local ``.env`` file). Model choice and API keys are not here: they
belong to aicore (``AICORE_*``), which makes every model call.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from text_to_sql_agent.agent import AgentLimits

# projects/03-text-to-sql-agent/ (this file is src/text_to_sql_agent/config.py).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TextToSqlSettings(BaseSettings):
    """Settings for the databases, query limits, agent budgets, the API and evals.

    Example: ``TEXT_TO_SQL_MAX_ROWS=50`` sets ``max_rows`` to 50.
    """

    model_config = SettingsConfigDict(
        env_prefix="TEXT_TO_SQL_",
        env_file=".env",
        extra="ignore",
    )

    # The agent's connection: the read-only role on the chinook database.
    # `load-db` creates this role with the user and password given here.
    database_url: str = "postgresql://sql_agent_ro:sql_agent_ro@localhost:5432/chinook"
    # Conversation memory (LangGraph checkpoints), used by the HTTP API.
    # A separate database and read-write role: never the agent's SQL role.
    # `init-memory` creates both from this URL.
    memory_database_url: str = (
        "postgresql://sql_agent_app:sql_agent_app@localhost:5432/text_to_sql_memory"
    )
    # Owner connection used only by `load-db` to create the database and
    # the role. Any existing database works (`postgres` always exists).
    admin_database_url: str = "postgresql://postgres:postgres@localhost:5432/postgres"
    # Vendored Chinook script (MIT licence, see data/chinook/LICENSE.md).
    chinook_sql: Path = _PROJECT_ROOT / "data" / "chinook" / "Chinook_PostgreSql.sql"
    # Golden dataset (golden.jsonl) and saved eval results (results/).
    evals_dir: Path = _PROJECT_ROOT / "evals"
    # Per-query cap enforced by Postgres. Chinook queries take milliseconds;
    # this only stops runaway cross joins.
    statement_timeout_ms: int = Field(default=10_000, gt=0)
    # Rows returned to the caller; more are cut off and flagged truncated,
    # so a huge result cannot flood the model's context or the API.
    max_rows: int = Field(default=200, gt=0, le=10_000)
    # Agent budgets (see AgentLimits). SQL-writing calls per turn: the first
    # try plus repairs. 4 of 5 baseline errors named their own fix, so one or
    # two repairs is where the value is; more mostly burns calls.
    max_attempts: int = Field(default=3, ge=1, le=5)
    # Earlier turns replayed into the prompt. Follow-ups refer to the last
    # turn or two; a fixed window keeps the prompt size bounded.
    history_turns: int = Field(default=3, ge=0, le=10)

    # HTTP API. A turn is 1-4 model calls: 20-90 s on a local 3B CPU model,
    # a few seconds on a hosted one. Past this the API answers 504; keep it
    # below gunicorn's --timeout (150 s in the Dockerfile).
    request_timeout_s: float = Field(default=120.0, gt=0)
    # Connections per pool and per worker process; matches gunicorn's
    # 4 threads, so a request never waits for a connection.
    pool_max_size: int = Field(default=4, ge=1, le=50)

    def agent_limits(self) -> AgentLimits:
        """The agent budgets as one object."""
        return AgentLimits(max_attempts=self.max_attempts, history_turns=self.history_turns)
