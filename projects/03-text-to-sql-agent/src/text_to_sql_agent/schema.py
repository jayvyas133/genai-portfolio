"""Read the database schema the model writes SQL against.

The schema is introspected from the live database, not written by hand,
so the prompt can never drift from the real tables. It is read as the
agent's own read-only role, so the model only ever sees tables it is
allowed to query.

``pg_catalog`` is used instead of ``information_schema``: the latter
only shows constraints on tables the current role owns, and the agent's
role owns nothing, so foreign keys (which tell the model how to join)
would be missing.
"""

import psycopg
from pydantic import BaseModel, ConfigDict

_COLUMNS = b"""
SELECT c.relname, a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_attribute a ON a.attrelid = c.oid
WHERE n.nspname = 'public'
  AND c.relkind IN ('r', 'v', 'm')
  AND a.attnum > 0 AND NOT a.attisdropped
  AND has_table_privilege(c.oid, 'SELECT')
ORDER BY c.relname, a.attnum
"""

# Primary keys first ('p' > 'f'), then foreign keys, in a stable order.
_CONSTRAINTS = b"""
SELECT conrelid::regclass::text, pg_get_constraintdef(oid)
FROM pg_constraint
WHERE connamespace = 'public'::regnamespace AND contype IN ('p', 'f')
ORDER BY 1, contype DESC, conname
"""


class Column(BaseModel):
    """One column, with its type as Postgres prints it."""

    model_config = ConfigDict(frozen=True)

    name: str
    type: str
    not_null: bool


class Table(BaseModel):
    """One table: columns in order plus its key constraints as SQL text."""

    model_config = ConfigDict(frozen=True)

    name: str
    columns: list[Column]
    constraints: list[str]

    def ddl(self) -> str:
        """The table as a ``CREATE TABLE`` statement, the form models know best."""
        lines = [f"  {c.name} {c.type}{' NOT NULL' if c.not_null else ''}" for c in self.columns]
        lines += [f"  {c}" for c in self.constraints]
        return f"CREATE TABLE {self.name} (\n" + ",\n".join(lines) + "\n);"


class Schema(BaseModel):
    """Every table the agent may query."""

    model_config = ConfigDict(frozen=True)

    tables: list[Table]

    def ddl(self) -> str:
        """All tables as DDL (Data Definition Language), separated by blank lines."""
        return "\n\n".join(t.ddl() for t in self.tables)


async def load_schema(database_url: str) -> Schema:
    """Introspect the ``public`` schema visible to the role in ``database_url``.

    Raises:
        psycopg.Error: If the database cannot be reached or queried.
    """
    async with await psycopg.AsyncConnection.connect(database_url) as conn:
        columns = await (await conn.execute(_COLUMNS)).fetchall()
        constraints = await (await conn.execute(_CONSTRAINTS)).fetchall()
        await conn.rollback()
    by_table: dict[str, list[Column]] = {}
    for table, name, type_, not_null in columns:
        by_table.setdefault(table, []).append(Column(name=name, type=type_, not_null=not_null))
    keys: dict[str, list[str]] = {}
    for table, definition in constraints:
        keys.setdefault(table, []).append(definition)
    return Schema(
        tables=[
            Table(name=name, columns=cols, constraints=keys.get(name, []))
            for name, cols in by_table.items()
        ]
    )
