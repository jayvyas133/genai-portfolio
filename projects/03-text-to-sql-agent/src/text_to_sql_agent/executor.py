"""Run one read-only query and return its rows.

Three layers keep model-written SQL harmless, each enforced by a
different component:

1. **The database role** (``database.py``): SELECT-only grants and
   ``default_transaction_read_only``. Postgres enforces it whatever the
   SQL says; this is the security boundary.
2. **The guard** (``guard.py``): one SELECT statement, nothing that
   writes or locks. It gives the model a clear error early.
3. **This module**: every query runs in a transaction marked READ ONLY
   with a ``statement_timeout``, is always rolled back, and returns at
   most ``max_rows`` rows.
"""

import time
from datetime import date, datetime
from datetime import time as dt_time
from decimal import Decimal
from typing import Any, Protocol

import psycopg
from psycopg import sql
from pydantic import BaseModel, ConfigDict, Field

from text_to_sql_agent.guard import check_sql

Value = str | int | float | bool | None
"""A cell as returned to callers: JSON-friendly, so results can be saved and served."""


class QueryResult(BaseModel):
    """The rows a query returned, cut off at ``max_rows``."""

    model_config = ConfigDict(frozen=True)

    columns: list[str]
    rows: list[list[Value]]
    truncated: bool = False
    elapsed_ms: float = Field(default=0.0, ge=0)


class SqlExecutionError(Exception):
    """Postgres rejected or cancelled the query.

    The message is Postgres's own (with its SQLSTATE code), because that is
    what a model needs to repair the query.
    """

    def __init__(self, message: str, sqlstate: str | None = None) -> None:
        """Keep Postgres's message and SQLSTATE code."""
        super().__init__(message)
        self.sqlstate = sqlstate


class QueryRunner(Protocol):
    """Anything that runs guarded, read-only SQL (``SqlExecutor``, or a fake in tests)."""

    async def run(self, text: str) -> QueryResult:
        """Run ``text`` and return its rows.

        Raises:
            UnsafeSqlError: If the guard rejects the SQL.
            SqlExecutionError: If the database rejects the query.
        """
        ...


def to_value(cell: Any) -> Value:
    """Convert a Postgres value to a JSON-friendly one.

    ``NUMERIC`` arrives as ``Decimal``; it becomes ``float``, which is
    exact enough for money with two decimals. Dates become ISO 8601 text.
    """
    if cell is None or isinstance(cell, str | bool | int | float):
        return cell
    if isinstance(cell, Decimal):
        return float(cell)
    if isinstance(cell, datetime | date | dt_time):
        return cell.isoformat()
    return str(cell)


class SqlExecutor:
    """Runs guarded, read-only queries against one database.

    A new connection per query keeps this simple and safe for the CLI and
    the eval. The HTTP API (M3) will share a connection pool instead.
    """

    def __init__(self, database_url: str, statement_timeout_ms: int, max_rows: int) -> None:
        """Connect with ``database_url`` (the read-only role) for each query."""
        self._url = database_url
        self._timeout_ms = statement_timeout_ms
        self._max_rows = max_rows

    async def run(self, text: str) -> QueryResult:
        """Check ``text`` with the guard, run it read-only and return its rows.

        Raises:
            UnsafeSqlError: If the guard rejects the SQL (nothing is run).
            SqlExecutionError: If Postgres rejects the query or it times out.
        """
        return await self.run_unguarded(check_sql(text))

    async def run_unguarded(self, text: str) -> QueryResult:
        """Run ``text`` without the guard; the database's own limits still apply.

        Only for tests that prove the database layer holds on its own.
        Everything else calls ``run``.

        Raises:
            SqlExecutionError: If Postgres rejects the query or it times out.
            psycopg.OperationalError: If the database cannot be reached;
                that is an outage, not a problem with the query.
        """
        start = time.perf_counter()
        async with await psycopg.AsyncConnection.connect(self._url) as conn:
            # Explicit, so these limits hold even if the role's defaults
            # were never set (layer 3 does not trust layer 1).
            await conn.set_read_only(True)
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        sql.SQL("SET LOCAL statement_timeout = {}").format(
                            sql.Literal(f"{self._timeout_ms}ms")
                        )
                    )
                    # bytes: psycopg's typing only accepts str literals,
                    # and this SQL is dynamic by nature (it was guarded).
                    await cur.execute(text.encode())
                    columns = [c.name for c in cur.description or []]
                    rows = await cur.fetchmany(self._max_rows + 1) if cur.description else []
            except psycopg.Error as err:
                diag = err.diag
                message = diag.message_primary or str(err)
                raise SqlExecutionError(
                    f"{message} (SQLSTATE {diag.sqlstate})", diag.sqlstate
                ) from err
            finally:
                # Nothing a read does needs keeping.
                await conn.rollback()
        return QueryResult(
            columns=columns,
            rows=[[to_value(c) for c in row] for row in rows[: self._max_rows]],
            truncated=len(rows) > self._max_rows,
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )
