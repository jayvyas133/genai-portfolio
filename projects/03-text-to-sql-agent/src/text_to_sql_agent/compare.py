"""Execution match: does the predicted query return the same answer as the gold one?

This is execution accuracy, the metric of the Spider and BIRD Text-to-SQL
benchmarks: different SQL that returns the same rows is equally right,
so results are compared, not query text. Three relaxations make it fit
a chatbot, where a user reads the table rather than a program:

- **Column order does not matter**, and column names are ignored.
- **Extra columns are allowed.** "Which genre has the most tracks?"
  answered with the genre *and* its count is a good answer. Every gold
  column must still be present.
- **Numbers are compared to 2 decimal places**, so ``ROUND(AVG(x), 2)``
  matches ``AVG(x)`` and the integer ``2023`` matches ``2023.0``.

Row order matters only when the gold query sorts at the top level (the
caller decides, see ``guard.has_order_by``). Row counts must match
exactly.
"""

from collections import Counter
from collections.abc import Sequence
from itertools import product

from text_to_sql_agent.executor import QueryResult, Value

# Column assignments tried before giving up. Real answers have a handful
# of columns, so this is only a guard against pathological SELECT *.
_MAX_ASSIGNMENTS = 10_000

Cell = str | float | bool | None


def _cell(value: Value) -> Cell:
    """Normalize one value: every number becomes a float rounded to 2 decimals."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    return round(float(value), 2)


def _column(rows: Sequence[Sequence[Cell]], index: int) -> Counter[Cell]:
    """The multiset of values in one column."""
    return Counter(row[index] for row in rows)


def results_match(predicted: QueryResult, gold: QueryResult, *, ordered: bool) -> bool:
    """Whether ``predicted`` answers the same as ``gold``, under the relaxations above.

    For each gold column, the candidates are the predicted columns holding
    the same multiset of values. Each one-to-one choice of candidates is
    then checked row by row, so two columns with the same values in a
    different pairing do not match by accident.
    """
    if len(predicted.rows) != len(gold.rows) or len(predicted.columns) < len(gold.columns):
        return False
    pred = [[_cell(v) for v in row] for row in predicted.rows]
    want = [tuple(_cell(v) for v in row) for row in gold.rows]
    if not want:
        return True
    pred_columns = [_column(pred, i) for i in range(len(predicted.columns))]
    candidates = []
    for j in range(len(gold.columns)):
        values = _column(want, j)
        matching = [i for i, column in enumerate(pred_columns) if column == values]
        if not matching:
            return False
        candidates.append(matching)

    target: list[tuple[Cell, ...]] | Counter[tuple[Cell, ...]] = want if ordered else Counter(want)
    for tried, assignment in enumerate(product(*candidates)):
        if tried >= _MAX_ASSIGNMENTS:
            break
        if len(set(assignment)) < len(assignment):
            continue  # one predicted column cannot stand in for two gold columns
        projected = [tuple(row[i] for i in assignment) for row in pred]
        if (projected if ordered else Counter(projected)) == target:
            return True
    return False
