"""Tests for the command line's parser and table output."""

import pytest
from text_to_sql_agent.cli import build_parser, format_table
from text_to_sql_agent.executor import QueryResult


def test_table_pads_columns_and_shows_null() -> None:
    result = QueryResult(columns=["name", "n"], rows=[["Rock", 1297], [None, 3]])
    assert format_table(result) == "name | n   \n-----+-----\nRock | 1297\nNULL | 3   "


def test_table_says_how_many_rows_are_hidden_and_if_truncated() -> None:
    result = QueryResult(columns=["x"], rows=[[i] for i in range(5)], truncated=True)
    text = format_table(result, limit=2)
    assert text.endswith("... 3 more row(s) shown of 5 (truncated by the row limit)")


def test_parser_defaults_and_choices() -> None:
    args = build_parser().parse_args(["eval", "--split", "dev", "--limit", "3"])
    assert (args.agent, args.split, args.limit) == ("single-shot", "dev", 3)
    assert build_parser().parse_args(["load-db", "--reset"]).reset
    with pytest.raises(SystemExit):
        build_parser().parse_args(["eval", "--agent", "nope"])
