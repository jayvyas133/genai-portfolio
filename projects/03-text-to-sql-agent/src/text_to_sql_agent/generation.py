"""The SQL-writing prompt and how SQL is read back out of the reply.

The reply is plain text with the query in a fenced code block, not JSON:
SQL is full of quotes and newlines, which small models often escape
wrongly inside a JSON string. A fence needs no escaping at all.
"""

import re

from aicore.models import LLMRequest, Message

SQL_PROMPT_VERSION = "v1"
"""Saved with eval results; bump it whenever the prompt changes."""

SYSTEM_PROMPT = """\
You translate questions about a PostgreSQL database into SQL.

Write one read-only PostgreSQL SELECT query that answers the question. \
Use only the tables and columns in the schema below. Select only the \
columns the question asks for, with clear names.

Reply with the query in a ```sql code block and nothing else.

Database schema:

{schema}"""

_FENCE = re.compile(r"```(?:sql|postgresql|postgres)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_BARE = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)


def build_sql_request(schema_ddl: str, question: str) -> LLMRequest:
    """One request asking for the SQL that answers ``question``.

    Temperature 0 makes the answer repeatable (and cacheable). 512 tokens
    is far more than any query over this schema needs.
    """
    return LLMRequest(
        messages=[
            Message(role="system", content=SYSTEM_PROMPT.format(schema=schema_ddl)),
            Message(role="user", content=question),
        ],
        temperature=0.0,
        max_tokens=512,
    )


def extract_sql(reply: str) -> str | None:
    """The SQL in ``reply``: the first fenced block, else the whole reply if it is a query.

    Returns ``None`` when the reply holds no SQL (for example a refusal or
    an explanation), so the caller can say so instead of running prose.
    """
    match = _FENCE.search(reply)
    if match:
        return match.group(1).strip() or None
    if _BARE.match(reply):
        return reply.strip()
    return None
