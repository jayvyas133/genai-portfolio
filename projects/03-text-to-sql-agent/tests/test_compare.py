"""Tests for execution match: same answer, regardless of how it is written."""

from text_to_sql_agent.compare import results_match
from text_to_sql_agent.executor import QueryResult, Value


def _r(columns: list[str], *rows: list[Value]) -> QueryResult:
    return QueryResult(columns=columns, rows=list(rows))


GOLD = _r(["country", "revenue"], ["USA", 523.06], ["Canada", 303.96])


def test_identical_results_match() -> None:
    assert results_match(GOLD, GOLD, ordered=False)


def test_row_order_is_ignored_unless_ordered() -> None:
    swapped = _r(["country", "revenue"], ["Canada", 303.96], ["USA", 523.06])
    assert results_match(swapped, GOLD, ordered=False)
    assert not results_match(swapped, GOLD, ordered=True)


def test_column_order_and_names_are_ignored() -> None:
    other = _r(["total", "billing_country"], [523.06, "USA"], [303.96, "Canada"])
    assert results_match(other, GOLD, ordered=True)


def test_extra_columns_are_allowed_but_missing_ones_are_not() -> None:
    extra = _r(["id", "country", "revenue"], [1, "USA", 523.06], [2, "Canada", 303.96])
    assert results_match(extra, GOLD, ordered=True)
    missing = _r(["country"], ["USA"], ["Canada"])
    assert not results_match(missing, GOLD, ordered=False)


def test_numbers_compare_to_two_decimals_and_across_int_and_float() -> None:
    gold = _r(["year", "avg"], [2023, 5.651428571])
    assert results_match(_r(["y", "a"], [2023.0, 5.65]), gold, ordered=False)
    assert not results_match(_r(["y", "a"], [2023, 5.66]), gold, ordered=False)


def test_row_count_must_match() -> None:
    assert not results_match(_r(["country", "revenue"], ["USA", 523.06]), GOLD, ordered=False)


def test_columns_with_the_same_values_must_still_pair_up_row_by_row() -> None:
    # Each column holds {a, b} in both, but the pairing differs.
    gold = _r(["x", "y"], ["a", "a"], ["b", "b"])
    crossed = _r(["x", "y"], ["a", "b"], ["b", "a"])
    assert not results_match(crossed, gold, ordered=False)


def test_duplicate_rows_count_as_a_multiset() -> None:
    gold = _r(["name"], ["Music"], ["Music"], ["Movies"])
    assert results_match(_r(["n"], ["Movies"], ["Music"], ["Music"]), gold, ordered=False)
    assert not results_match(_r(["n"], ["Movies"], ["Movies"], ["Music"]), gold, ordered=False)


def test_null_matches_only_null() -> None:
    gold = _r(["company"], [None])
    assert results_match(_r(["c"], [None]), gold, ordered=False)
    assert not results_match(_r(["c"], ["None"]), gold, ordered=False)
