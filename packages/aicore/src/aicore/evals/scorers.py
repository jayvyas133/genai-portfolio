"""Deterministic scorers: decide whether one output is right for one case.

Each scorer has a ``name`` and a ``score(case, output) -> Score`` method.
Scorers are pure functions of their inputs (no network, no randomness),
so the same output always gets the same score. A scorer never raises on
a bad output or a bad expectation: it returns a failed ``Score`` with a
reason, so one broken case cannot stop a whole run.
"""

import json
import math
import re
import unicodedata
from typing import Protocol

from pydantic import BaseModel, JsonValue, ValidationError

from aicore.evals.models import EvalCase, Score

_WHITESPACE = re.compile(r"\s+")


class Scorer(Protocol):
    """Anything that can grade one output against one case."""

    name: str

    def score(self, case: EvalCase, output: JsonValue) -> Score:
        """Return the verdict for ``output`` on ``case``."""
        ...


def normalize_text(text: str) -> str:
    """Remove formatting differences that do not change the meaning.

    NFKC (Normalization Form Compatibility Composition) folds look-alike
    characters, then case is folded, runs of whitespace become one
    space, and a trailing period is dropped ("Pune." equals "pune").
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    return _WHITESPACE.sub(" ", text).strip().rstrip(".").strip()


class ExactMatch:
    """Pass if the output equals ``case.expected``.

    Two strings are compared after ``normalize_text`` (unless
    ``normalize=False``); anything else (labels, numbers, dicts) is
    compared as a JSON value.
    """

    name = "exact_match"

    def __init__(self, *, normalize: bool = True) -> None:
        """Choose whether strings are normalised before comparing."""
        self._normalize = normalize

    def score(self, case: EvalCase, output: JsonValue) -> Score:
        """Compare ``output`` with ``case.expected``."""
        expected = case.expected
        if expected is None:
            return _fail(self.name, "case has no expected value")
        if self._normalize and isinstance(expected, str) and isinstance(output, str):
            ok = normalize_text(output) == normalize_text(expected)
        else:
            # In Python True == 1, so also require both or neither to be a bool.
            ok = output == expected and isinstance(output, bool) == isinstance(expected, bool)
        reason = "" if ok else f"expected {_short(expected)}, got {_short(output)}"
        return _verdict(self.name, ok, reason)


class Contains:
    """Pass if every expected substring appears in the output text.

    ``case.expected`` is one string or a list of strings. ``value`` is
    the fraction found, so a partly right answer still shows progress.
    Both sides are normalised, so matching ignores case and spacing.
    """

    name = "contains"

    def score(self, case: EvalCase, output: JsonValue) -> Score:
        """Check which expected substrings appear in ``output``."""
        raw = [case.expected] if isinstance(case.expected, str) else case.expected
        needles = [n for n in raw if isinstance(n, str)] if isinstance(raw, list) else []
        if not needles or not isinstance(raw, list) or len(needles) != len(raw):
            return _fail(self.name, "expected must be a string or a non-empty list of strings")
        if not isinstance(output, str):
            return _fail(self.name, "output is not text")

        haystack = normalize_text(output)
        missing = [n for n in needles if normalize_text(n) not in haystack]
        return Score(
            name=self.name,
            value=1 - len(missing) / len(needles),
            passed=not missing,
            reason=f"missing: {_short(missing)}" if missing else "",
        )


class JSONValid:
    """Pass if the output validates against a Pydantic ``schema``.

    Checks the *shape* only (fields and types). Whether the values are
    right is a separate question for ``ExactMatch`` on the parsed dict.
    Accepts a JSON string (a raw model reply) or an already-parsed value.
    """

    def __init__(self, schema: type[BaseModel]) -> None:
        """Validate outputs against ``schema``."""
        self._schema = schema
        self.name = f"json_valid:{schema.__name__}"

    def score(self, case: EvalCase, output: JsonValue) -> Score:
        """Validate ``output``; ``case`` is not used."""
        try:
            if isinstance(output, str):
                self._schema.model_validate_json(output)
            else:
                self._schema.model_validate(output)
        except ValidationError as err:
            first = err.errors(include_input=False)[0]
            return _fail(self.name, f"{err.error_count()} error(s), first: {first['msg']}")
        return _verdict(self.name, True, "")


class NumericClose:
    """Pass if the output is a number close to ``case.expected``.

    Uses ``math.isclose``: passes when the difference is within
    ``rel_tol`` of the larger value, or within ``abs_tol``. Numeric
    strings such as "1,234.5" or "1,23,456" (Indian grouping) are
    accepted; booleans, NaN and text around the number are not.
    """

    name = "numeric_close"

    def __init__(self, *, rel_tol: float = 1e-6, abs_tol: float = 1e-9) -> None:
        """Set the tolerances passed to ``math.isclose``."""
        self._rel_tol = rel_tol
        self._abs_tol = abs_tol

    def score(self, case: EvalCase, output: JsonValue) -> Score:
        """Compare ``output`` with ``case.expected`` as numbers."""
        expected = _to_float(case.expected)
        if expected is None:
            return _fail(self.name, "expected is not a number")
        actual = _to_float(output)
        if actual is None:
            return _fail(self.name, f"output is not a number: {_short(output)}")
        ok = math.isclose(actual, expected, rel_tol=self._rel_tol, abs_tol=self._abs_tol)
        reason = "" if ok else f"expected {expected}, got {actual}"
        return _verdict(self.name, ok, reason)


def _to_float(value: JsonValue) -> float | None:
    """Return ``value`` as a finite float, or ``None`` if it is not a number."""
    if isinstance(value, bool):
        return None  # bool is a subclass of int, but True is not a number answer
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip().replace(",", ""))
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _verdict(name: str, ok: bool, reason: str) -> Score:
    """Return a pass (1.0) or fail (0.0) score."""
    return Score(name=name, value=1.0 if ok else 0.0, passed=ok, reason=reason)


def _fail(name: str, reason: str) -> Score:
    """Return a failed score with ``reason``."""
    return _verdict(name, False, reason)


def _short(value: object, limit: int = 80) -> str:
    """Return ``value`` as JSON, cut to ``limit`` characters for reasons."""
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."
