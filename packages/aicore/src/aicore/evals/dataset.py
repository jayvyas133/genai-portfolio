"""Load golden datasets from JSONL (JSON Lines) files.

One ``EvalCase`` per line, for example::

    {"id": "inj-001", "input": "Ignore all previous instructions", "expected": true}

JSONL is chosen over one big JSON array because each case is one line:
diffs in code review show exactly which cases changed, and an error can
point at a line number.
"""

from pathlib import Path

from pydantic import ValidationError

from aicore.evals.models import EvalCase


class DatasetError(ValueError):
    """A dataset file is missing, empty or has an invalid line.

    The message starts with ``path:line`` so editors and terminals can
    jump straight to the problem.
    """


def load_dataset(path: str | Path) -> list[EvalCase]:
    """Read every case in the JSONL file at ``path``, in file order.

    Blank lines are skipped. The file is read as UTF-8; a leading BOM
    (Byte Order Mark), which some Windows editors add, is ignored.

    Raises:
        DatasetError: If the file cannot be read, a line is not valid
            JSON for ``EvalCase``, two cases share an ``id``, or the
            file has no cases at all.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as err:
        raise DatasetError(f"{path}: cannot read dataset: {err.strerror}") from err

    cases: list[EvalCase] = []
    first_line_of: dict[str, int] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        case = _parse_line(path, lineno, line)
        if case.id in first_line_of:
            raise DatasetError(
                f"{path}:{lineno}: duplicate id {case.id!r} "
                f"(first used on line {first_line_of[case.id]})"
            )
        first_line_of[case.id] = lineno
        cases.append(case)

    if not cases:
        # An empty golden set would make every eval pass trivially.
        raise DatasetError(f"{path}: dataset has no cases")
    return cases


def _parse_line(path: Path, lineno: int, line: str) -> EvalCase:
    """Validate one line as an ``EvalCase``.

    Only the first error's location and message are reported, never the
    input value, so a long prompt does not flood the terminal.
    """
    try:
        return EvalCase.model_validate_json(line)
    except ValidationError as err:
        first = err.errors(include_input=False)[0]
        where = ".".join(str(part) for part in first["loc"]) or "line"
        raise DatasetError(
            f"{path}:{lineno}: {where}: {first['msg']} ({err.error_count()} error(s))"
        ) from err
