"""Tests for loading JSONL golden datasets."""

from pathlib import Path

import pytest
from aicore.evals.dataset import DatasetError, load_dataset


def write(tmp_path: Path, text: str, encoding: str = "utf-8") -> Path:
    """Write ``text`` to a dataset file in ``tmp_path`` and return its path."""
    path = tmp_path / "cases.jsonl"
    path.write_text(text, encoding=encoding)
    return path


def test_loads_cases_in_file_order(tmp_path: Path) -> None:
    """Each line becomes one case; order is kept for stable reports."""
    path = write(
        tmp_path,
        '{"id": "a", "input": "x", "expected": true}\n'
        '{"id": "b", "input": "y", "expected": {"city": "Pune"}, "tags": ["hi"]}\n',
    )
    cases = load_dataset(path)
    assert [c.id for c in cases] == ["a", "b"]
    assert cases[1].expected == {"city": "Pune"}
    assert cases[1].tags == ["hi"]


def test_skips_blank_lines(tmp_path: Path) -> None:
    """Blank lines (including a trailing newline) are not cases."""
    path = write(tmp_path, '\n{"id": "a", "input": "x"}\n   \n')
    assert len(load_dataset(path)) == 1


def test_reads_hindi_and_bom(tmp_path: Path) -> None:
    """UTF-8 text such as Hindi loads, and a Windows BOM is ignored."""
    path = write(tmp_path, '{"id": "a", "input": "नमस्ते"}\n', encoding="utf-8-sig")
    assert load_dataset(path)[0].input == "नमस्ते"


def test_invalid_json_names_the_line(tmp_path: Path) -> None:
    """A broken line is reported with its line number."""
    path = write(tmp_path, '{"id": "a", "input": "x"}\n{"id": "b", \n')
    with pytest.raises(DatasetError, match=r"cases\.jsonl:2:"):
        load_dataset(path)


def test_unknown_field_names_the_field(tmp_path: Path) -> None:
    """A typo in a field name fails and says which field."""
    path = write(tmp_path, '{"id": "a", "input": "x", "expcted": 1}\n')
    with pytest.raises(DatasetError, match=r":1: expcted:"):
        load_dataset(path)


def test_duplicate_id_names_both_lines(tmp_path: Path) -> None:
    """Results are keyed by id, so ids must be unique."""
    path = write(tmp_path, '{"id": "a", "input": "x"}\n{"id": "a", "input": "y"}\n')
    with pytest.raises(DatasetError, match=r":2: duplicate id 'a' \(first used on line 1\)"):
        load_dataset(path)


def test_empty_file_is_an_error(tmp_path: Path) -> None:
    """An empty golden set would make every eval pass trivially."""
    path = write(tmp_path, "\n\n")
    with pytest.raises(DatasetError, match="no cases"):
        load_dataset(path)


def test_missing_file_is_a_dataset_error(tmp_path: Path) -> None:
    """A wrong path gives the same error type as a bad line."""
    with pytest.raises(DatasetError, match="cannot read dataset"):
        load_dataset(tmp_path / "nope.jsonl")
