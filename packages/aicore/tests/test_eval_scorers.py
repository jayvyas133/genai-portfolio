"""Tests for the deterministic evaluation scorers."""

import pytest
from aicore.evals.models import EvalCase
from aicore.evals.scorers import (
    Contains,
    ExactMatch,
    JSONValid,
    NumericClose,
    Scorer,
    normalize_text,
)
from pydantic import BaseModel, JsonValue


class City(BaseModel):
    """Small schema for the ``JSONValid`` tests."""

    city: str
    population: int


def case(expected: JsonValue) -> EvalCase:
    """Return a case whose only interesting field is ``expected``."""
    return EvalCase(id="c1", input="question", expected=expected)


# --- normalize_text ---------------------------------------------------------


def test_normalize_folds_case_space_and_trailing_period() -> None:
    """Formatting differences that do not change meaning are removed."""
    assert normalize_text("  The   Capital is PUNE. ") == "the capital is pune"


def test_normalize_folds_full_width_characters() -> None:
    """NFKC turns full-width letters into plain ones."""
    assert normalize_text("ＰＵＮＥ") == "pune"


# --- ExactMatch -------------------------------------------------------------


def test_exact_match_ignores_formatting() -> None:
    """Case, spacing and a trailing period do not fail a correct answer."""
    score = ExactMatch().score(case("Pune"), "  pune. ")
    assert score.passed
    assert score.value == 1.0
    assert score.name == "exact_match"


def test_exact_match_strict_mode_compares_raw_text() -> None:
    """With normalisation off, any difference fails."""
    score = ExactMatch(normalize=False).score(case("Pune"), "pune")
    assert not score.passed
    assert score.value == 0.0
    assert "expected" in score.reason


def test_exact_match_compares_labels_and_dicts() -> None:
    """Non-text expectations are compared as JSON values."""
    assert ExactMatch().score(case(True), True).passed
    assert ExactMatch().score(case({"city": "Pune"}), {"city": "Pune"}).passed
    assert not ExactMatch().score(case({"city": "Pune"}), {"city": "Mumbai"}).passed


def test_exact_match_does_not_treat_true_as_one() -> None:
    """In Python ``True == 1``; for labels that would be a false pass."""
    assert not ExactMatch().score(case(True), 1).passed


def test_exact_match_fails_without_expected() -> None:
    """A case with nothing to compare against is a dataset bug."""
    score = ExactMatch().score(case(None), "anything")
    assert not score.passed
    assert "no expected" in score.reason


# --- Contains ---------------------------------------------------------------


def test_contains_single_string() -> None:
    """A string expectation must appear somewhere in the output."""
    assert Contains().score(case("pune"), "The capital is Pune.").passed


def test_contains_gives_partial_credit() -> None:
    """``value`` is the fraction found; ``passed`` needs all of them."""
    score = Contains().score(case(["Pune", "Mumbai"]), "Only Pune is listed")
    assert score.value == 0.5
    assert not score.passed
    assert "Mumbai" in score.reason


def test_contains_rejects_bad_expected() -> None:
    """Expected must be a string or a non-empty list of strings."""
    assert not Contains().score(case([]), "text").passed
    assert not Contains().score(case([1, 2]), "text").passed


def test_contains_needs_text_output() -> None:
    """A dict or number output cannot contain a substring."""
    score = Contains().score(case("pune"), {"city": "pune"})
    assert not score.passed
    assert "not text" in score.reason


# --- JSONValid --------------------------------------------------------------


def test_json_valid_accepts_json_string_and_dict() -> None:
    """Both a raw model reply and an already-parsed dict are checked."""
    scorer = JSONValid(City)
    assert scorer.name == "json_valid:City"
    assert scorer.score(case(None), '{"city": "Pune", "population": 7000000}').passed
    assert scorer.score(case(None), {"city": "Pune", "population": 7000000}).passed


def test_json_valid_rejects_wrong_shape() -> None:
    """A missing field fails and the reason counts the errors."""
    score = JSONValid(City).score(case(None), '{"city": "Pune"}')
    assert not score.passed
    assert "1 error(s)" in score.reason


def test_json_valid_rejects_non_json_text() -> None:
    """Prose instead of JSON is a failure, not a crash."""
    assert not JSONValid(City).score(case(None), "Sure! Here is the JSON").passed


# --- NumericClose -----------------------------------------------------------


@pytest.mark.parametrize(
    ("expected", "output"),
    [
        (42, 42),
        (42, 42.0000001),
        (1234.5, "1,234.5"),
        (123456, "1,23,456"),
        (0.25, " 0.25 "),
    ],
)
def test_numeric_close_passes(expected: JsonValue, output: JsonValue) -> None:
    """Numbers, numeric strings and thousands separators are accepted."""
    assert NumericClose().score(case(expected), output).passed


def test_numeric_close_respects_tolerance() -> None:
    """``rel_tol`` decides how far off an answer may be."""
    assert not NumericClose().score(case(100), 101).passed
    assert NumericClose(rel_tol=0.02).score(case(100), 101).passed


@pytest.mark.parametrize("output", ["about 42", True, None, "nan", {"n": 42}])
def test_numeric_close_rejects_non_numbers(output: JsonValue) -> None:
    """Text, booleans, NaN and objects are not numbers."""
    score = NumericClose().score(case(42), output)
    assert not score.passed
    assert "not a number" in score.reason


def test_numeric_close_needs_numeric_expected() -> None:
    """A non-numeric expectation is a dataset bug."""
    score = NumericClose().score(case("forty-two"), 42)
    assert not score.passed
    assert "expected" in score.reason


# --- protocol ---------------------------------------------------------------


def test_all_scorers_satisfy_the_protocol() -> None:
    """Every built-in scorer can be used wherever a ``Scorer`` is expected."""
    scorers: list[Scorer] = [ExactMatch(), Contains(), JSONValid(City), NumericClose()]
    names = [s.name for s in scorers]
    assert names == ["exact_match", "contains", "json_valid:City", "numeric_close"]
