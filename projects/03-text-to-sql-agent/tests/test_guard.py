"""Tests for the SQL guard: what counts as one safe, read-only query."""

import pytest
from text_to_sql_agent.guard import UnsafeSqlError, check_sql, has_order_by


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "select name from genre where name = 'delete'",
        "SELECT count(*) FROM track;",
        "WITH t AS (SELECT * FROM track) SELECT count(*) FROM t",
        "SELECT name FROM genre UNION SELECT name FROM media_type",
        "SELECT title FROM album a WHERE EXISTS (SELECT 1 FROM track t WHERE t.album_id = 1)",
        'SELECT "update" FROM (SELECT 1 AS "update") s',
    ],
)
def test_read_only_queries_pass(sql: str) -> None:
    assert check_sql(sql) == sql.strip().rstrip(";")


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("DELETE FROM genre", "Only SELECT queries are allowed, not DELETE"),
        ("UPDATE track SET unit_price = 0", "not UPDATE"),
        ("INSERT INTO genre VALUES (99, 'x')", "not INSERT"),
        ("DROP TABLE genre", "not DROP"),
        ("TRUNCATE genre", "not TRUNCATE"),
        ("CREATE TABLE x (i int)", "not CREATE"),
        ("SET statement_timeout = 0", "not SET"),
        ("VACUUM", "not COMMAND"),
        ("SELECT 1; DROP TABLE genre", "exactly one statement, not 2"),
        ("WITH d AS (DELETE FROM genre RETURNING *) SELECT * FROM d", "found DELETE"),
        ("SELECT * INTO copy FROM genre", "found INTO"),
        ("SELECT * FROM genre FOR UPDATE", "found LOCK"),
        ("", "no SQL statement"),
        ("SELECT FROM WHERE", "does not parse"),
    ],
)
def test_writes_locks_and_garbage_are_rejected_with_a_reason(sql: str, reason: str) -> None:
    with pytest.raises(UnsafeSqlError, match=reason):
        check_sql(sql)


def test_has_order_by_only_counts_the_outer_query() -> None:
    assert has_order_by("SELECT name FROM genre ORDER BY name")
    assert has_order_by("SELECT name FROM genre UNION SELECT name FROM artist ORDER BY 1")
    assert not has_order_by("SELECT * FROM (SELECT name FROM genre ORDER BY name LIMIT 3) s")
    assert not has_order_by("SELECT count(*) FROM track")
