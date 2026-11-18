"""Tests that need no database: schema rendering, value conversion, loader checks."""

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from text_to_sql_agent.config import TextToSqlSettings
from text_to_sql_agent.database import LoadError, load_chinook, script_body
from text_to_sql_agent.executor import to_value
from text_to_sql_agent.schema import Column, Schema, Table


def test_table_renders_as_create_table_with_keys() -> None:
    table = Table(
        name="album",
        columns=[
            Column(name="album_id", type="integer", not_null=True),
            Column(name="title", type="character varying(160)", not_null=False),
        ],
        constraints=["PRIMARY KEY (album_id)"],
    )
    assert table.ddl() == (
        "CREATE TABLE album (\n"
        "  album_id integer NOT NULL,\n"
        "  title character varying(160),\n"
        "  PRIMARY KEY (album_id)\n"
        ");"
    )
    assert Schema(tables=[table, table]).ddl().count("\n\nCREATE TABLE") == 1


def test_postgres_values_become_json_friendly() -> None:
    assert to_value(Decimal("3.96")) == 3.96
    assert to_value(datetime(2024, 3, 1, 0, 0)) == "2024-03-01T00:00:00"
    assert to_value(date(2024, 3, 1)) == "2024-03-01"
    assert [to_value(v) for v in (None, True, 7, 1.5, "x")] == [None, True, 7, 1.5, "x"]


def test_script_body_skips_the_psql_only_header() -> None:
    body = script_body(TextToSqlSettings(_env_file=None).chinook_sql)
    assert "CREATE DATABASE" not in body and "\\c" not in body
    assert body.lstrip().startswith("/*") and "CREATE TABLE album" in body


def test_script_without_the_connect_line_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "x.sql"
    path.write_text("CREATE TABLE t (i int);", encoding="utf-8")
    with pytest.raises(LoadError, match="expected"):
        script_body(path)


@pytest.mark.parametrize(
    ("readonly_url", "message"),
    [
        ("postgresql://postgres:pw@localhost/chinook", "must not be the admin user"),
        ("postgresql://ro@localhost/chinook", "needs a user, a password and a database"),
    ],
)
def test_loader_refuses_unsafe_or_incomplete_role_settings(
    readonly_url: str, message: str, tmp_path: Path
) -> None:
    # Checked before any connection is made, so no database is needed.
    with pytest.raises(LoadError, match=message):
        load_chinook(
            "postgresql://postgres:pw@localhost/postgres",
            readonly_url,
            tmp_path / "unused.sql",
            1000,
            reset=False,
        )
