"""Tests for the SQL prompt and reading SQL back out of a reply."""

from text_to_sql_agent.generation import build_sql_request, extract_sql


def test_request_puts_the_schema_in_the_system_prompt_and_the_question_last() -> None:
    request = build_sql_request("CREATE TABLE genre (...);", "How many genres?")
    system, user = request.messages
    assert system.role == "system" and "CREATE TABLE genre (...);" in system.content
    assert user.role == "user" and user.content == "How many genres?"
    assert request.temperature == 0.0


def test_extracts_the_first_fenced_block() -> None:
    reply = "Here you go:\n```sql\nSELECT 1;\n```\nand another\n```sql\nSELECT 2\n```"
    assert extract_sql(reply) == "SELECT 1;"


def test_accepts_an_unlabelled_or_postgresql_fence() -> None:
    assert extract_sql("```\nSELECT 1\n```") == "SELECT 1"
    assert extract_sql("```PostgreSQL\nSELECT 1\n```") == "SELECT 1"


def test_a_bare_query_is_used_as_is() -> None:
    assert extract_sql("  WITH t AS (SELECT 1) SELECT * FROM t  ") == (
        "WITH t AS (SELECT 1) SELECT * FROM t"
    )


def test_prose_or_an_empty_fence_is_not_sql() -> None:
    assert extract_sql("I cannot answer that from this schema.") is None
    assert extract_sql("```sql\n\n```") is None
