"""Integration tests against the real Chinook database.

Run ``uv run text-to-sql load-db`` first (Postgres from infra/docker-compose.yml).
Each test proves one read-only layer holds *on its own*, with the
layers in front of it bypassed.
"""

import psycopg
import pytest
from text_to_sql_agent.config import TextToSqlSettings
from text_to_sql_agent.executor import SqlExecutionError, SqlExecutor
from text_to_sql_agent.golden import load_golden, run_gold
from text_to_sql_agent.guard import UnsafeSqlError
from text_to_sql_agent.schema import load_schema

pytestmark = pytest.mark.integration

SETTINGS = TextToSqlSettings(_env_file=None)


def _executor(timeout_ms: int = 2000, max_rows: int = 5) -> SqlExecutor:
    return SqlExecutor(SETTINGS.database_url, timeout_ms, max_rows)


async def test_rows_come_back_capped_and_flagged() -> None:
    result = await _executor().run("SELECT name FROM genre ORDER BY name")
    assert result.columns == ["name"]
    assert result.rows[0] == ["Alternative"] and len(result.rows) == 5
    assert result.truncated


async def test_guard_stops_a_write_before_the_database_sees_it() -> None:
    with pytest.raises(UnsafeSqlError):
        await _executor().run("DELETE FROM genre")


@pytest.mark.parametrize(
    "sql", ["DELETE FROM genre", "CREATE TEMP TABLE t (i int)", "SELECT * FROM genre FOR UPDATE"]
)
async def test_read_only_transaction_stops_writes_without_the_guard(sql: str) -> None:
    with pytest.raises(SqlExecutionError) as err:
        await _executor().run_unguarded(sql)
    assert err.value.sqlstate == "25006"  # read_only_sql_transaction


async def test_statement_timeout_cancels_slow_queries() -> None:
    with pytest.raises(SqlExecutionError) as err:
        await _executor(timeout_ms=200).run("SELECT pg_sleep(2)")
    assert err.value.sqlstate == "57014"  # query_canceled


async def test_grants_alone_stop_writes_even_with_read_only_mode_switched_off() -> None:
    # Bypass layers 2 and 3: a raw session of the agent's role that turns
    # off its read-only default. Only the grants are left.
    async with await psycopg.AsyncConnection.connect(
        SETTINGS.database_url, autocommit=True
    ) as conn:
        await conn.execute("SET default_transaction_read_only = off")
        for statement in (b"DELETE FROM genre", b"CREATE TABLE public.x (i int)"):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                await conn.execute(statement)


async def test_schema_shows_every_table_with_its_foreign_keys() -> None:
    schema = await load_schema(SETTINGS.database_url)
    assert len(schema.tables) == 11
    track = next(t for t in schema.tables if t.name == "track")
    assert "FOREIGN KEY (album_id) REFERENCES album(album_id)" in track.constraints


async def test_every_gold_query_runs_and_returns_rows() -> None:
    cases = load_golden(SETTINGS.evals_dir / "golden.jsonl")
    executor = SqlExecutor(SETTINGS.database_url, SETTINGS.statement_timeout_ms, SETTINGS.max_rows)
    results, problems = await run_gold(cases, executor)
    assert problems == []
    assert len(results) == sum(len(c.turns) for c in cases)
