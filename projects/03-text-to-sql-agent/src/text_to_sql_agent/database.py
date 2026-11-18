"""Create the Chinook sample database and the agent's read-only role.

This is an admin task, run once with ``text-to-sql load-db``. It connects
as the owner (``admin_database_url``), never as the agent. The role it
creates is the real security boundary of the system: whatever SQL the
model writes, Postgres itself refuses to let that role change anything.

``infra/postgres/init.sql`` is not used because Postgres only runs it on
the container's very first start, and the portfolio's volume already
exists.
"""

import logging
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

logger = logging.getLogger(__name__)

# The vendored script starts with DROP/CREATE DATABASE and the psql
# meta-command below, which only psql understands. Everything after it
# is plain SQL that psycopg can send in one call.
_CONNECT_LINE = "\\c chinook;"


class LoadError(Exception):
    """The database or role could not be set up as configured."""


def script_body(path: Path) -> str:
    """The table-creating part of the Chinook script at ``path``.

    Raises:
        LoadError: If the script does not have the expected layout.
    """
    text = path.read_text(encoding="utf-8")
    _, sep, body = text.partition(_CONNECT_LINE)
    if not sep:
        raise LoadError(f"{path}: expected a {_CONNECT_LINE!r} line")
    return body


def load_chinook(
    admin_url: str, readonly_url: str, script: Path, statement_timeout_ms: int, reset: bool
) -> bool:
    """Create the database named in ``readonly_url`` and its read-only role.

    Idempotent: an existing database is kept (unless ``reset``), and the
    role's password, settings and grants are always re-applied, so
    changing them in the environment and re-running is enough.

    Args:
        admin_url: Owner connection to any existing database.
        readonly_url: The agent's connection; its user, password and
            database name decide what is created.
        script: The Chinook PostgreSQL script.
        statement_timeout_ms: Also stored on the role, so even a
            connection that forgets to set it is capped.
        reset: Drop and reload the database.

    Returns:
        ``True`` if the data was (re)loaded, ``False`` if it already existed.

    Raises:
        LoadError: If the read-only URL names the admin user or lacks a
            user, password or database.
        psycopg.Error: If Postgres rejects a statement.
    """
    ro = conninfo_to_dict(readonly_url)
    admin = conninfo_to_dict(admin_url)
    role, password, dbname = (str(ro.get(key) or "") for key in ("user", "password", "dbname"))
    if not (role and password and dbname):
        raise LoadError("TEXT_TO_SQL_DATABASE_URL needs a user, a password and a database")
    if role == admin.get("user"):
        raise LoadError("the read-only user must not be the admin user")

    loaded = False
    with psycopg.connect(admin_url, autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone()
        if exists and reset:
            conn.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(dbname)))
            exists = None
        if not exists:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
            loaded = True
        _ensure_role(conn, role, password, statement_timeout_ms)

    db_url = make_conninfo(admin_url, dbname=dbname)
    if loaded:
        try:
            # One transaction: a failed load leaves no half-filled tables.
            with psycopg.connect(db_url) as conn:
                conn.execute(script_body(script).encode())
        except BaseException:
            with psycopg.connect(admin_url, autocommit=True) as conn:
                conn.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(dbname)))
            raise
        logger.info("loaded %s into database %s", script.name, dbname)

    with psycopg.connect(db_url, autocommit=True) as conn:
        _grant_read_only(conn, dbname, role)
    return loaded


def _ensure_role(conn: psycopg.Connection, role: str, password: str, timeout_ms: int) -> None:
    """Create or update the login role with read-only defaults."""
    exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone()
    verb = sql.SQL("ALTER" if exists else "CREATE")
    name = sql.Identifier(role)
    # CREATE/ALTER ROLE take no bind parameters, so the password is quoted
    # as a literal by psycopg (never by string formatting).
    conn.execute(
        sql.SQL("{} ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD {}").format(
            verb, name, sql.Literal(password)
        )
    )
    # Every session of this role starts read-only and with a timeout.
    conn.execute(sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(name))
    conn.execute(
        sql.SQL("ALTER ROLE {} SET statement_timeout = {}").format(
            name, sql.Literal(f"{timeout_ms}ms")
        )
    )


def _grant_read_only(conn: psycopg.Connection, dbname: str, role: str) -> None:
    """Let ``role`` connect and SELECT, and nothing else."""
    db, name = sql.Identifier(dbname), sql.Identifier(role)
    for statement in (
        # PUBLIC (every role) gets CONNECT and TEMP by default; keep that
        # to the agent's role only, and without TEMP.
        sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC").format(db),
        sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(db, name),
        sql.SQL("REVOKE CREATE ON SCHEMA public FROM PUBLIC"),
        sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(name),
        sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {}").format(name),
    ):
        conn.execute(statement)
