"""Check model-written SQL before it reaches the database.

The database role is the security boundary (see ``database.py``): it is
read-only whatever this module decides. The guard is the second layer,
and it exists for two practical reasons:

- **A clear, early error.** "Only SELECT queries are allowed" is far more
  useful to the model when it repairs its query than Postgres's
  "cannot execute DELETE in a read-only transaction".
- **Defence in depth.** If the role were ever misconfigured, a write
  would still be stopped here.

It parses the SQL with sqlglot instead of matching keywords: a regex
cannot reliably tell a ``DELETE`` inside a CTE (Common Table Expression)
from the word "delete" inside a string literal or column name.
"""

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

# Allowed anywhere only as part of a read: any node of these types means
# the statement writes, changes the schema, or takes locks.
_FORBIDDEN = (
    exp.DML,  # INSERT, UPDATE, DELETE, MERGE, COPY (also inside a CTE)
    exp.DDL,  # CREATE ...
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Command,  # anything sqlglot does not model: VACUUM, GRANT, ...
    exp.Set,  # SET <setting>: could lift the statement timeout
    exp.Into,  # SELECT ... INTO new_table creates a table
    exp.Lock,  # SELECT ... FOR UPDATE / FOR SHARE takes row locks
)


class UnsafeSqlError(ValueError):
    """The SQL is not a single read-only query; the message says why."""


def check_sql(text: str) -> str:
    """Return ``text`` stripped of whitespace and a trailing ``;`` if it is safe to run.

    Safe means: it parses as PostgreSQL, it is exactly one statement, the
    statement is a query (SELECT, WITH ... SELECT, UNION, ...), and nothing
    inside it writes or locks.

    Raises:
        UnsafeSqlError: With a message written for the model to act on.
    """
    try:
        statements = [s for s in sqlglot.parse(text, read="postgres") if s is not None]
    except ParseError as err:
        raise UnsafeSqlError(f"The SQL does not parse: {_first_line(err)}") from err
    if not statements:
        raise UnsafeSqlError("There is no SQL statement.")
    if len(statements) > 1:
        raise UnsafeSqlError(f"Write exactly one statement, not {len(statements)}.")
    (statement,) = statements
    if not isinstance(statement, exp.Query):
        raise UnsafeSqlError(f"Only SELECT queries are allowed, not {statement.key.upper()}.")
    forbidden = statement.find(*_FORBIDDEN)
    if forbidden is not None:
        raise UnsafeSqlError(
            f"Only read-only queries are allowed; found {forbidden.key.upper()} in the query."
        )
    return text.strip().rstrip(";").strip()


def has_order_by(text: str) -> bool:
    """Whether the outermost query of ``text`` sorts its rows.

    The eval compares rows in order only when the gold query has an
    ORDER BY at the top level; an ORDER BY in a subquery does not fix the
    order of the final result.

    Raises:
        UnsafeSqlError: If ``text`` does not parse.
    """
    try:
        statement = sqlglot.parse_one(text, read="postgres")
    except ParseError as err:
        raise UnsafeSqlError(f"The SQL does not parse: {_first_line(err)}") from err
    return statement.args.get("order") is not None


def _first_line(err: ParseError) -> str:
    """sqlglot's message without the multi-line SQL excerpt it appends."""
    return str(err).splitlines()[0]
