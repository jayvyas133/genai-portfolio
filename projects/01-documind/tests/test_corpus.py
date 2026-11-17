"""Tests for ``load_documents`` and the committed sample corpus."""

import json
from pathlib import Path

import pytest
from documind.config import DocumindSettings
from documind.corpus import Document, load_documents
from pydantic import ValidationError


def write_doc(folder: Path, name: str, **fields: str) -> None:
    """Write one document JSON file into ``folder``."""
    doc = {"title": "T", "url": "https://example.org/t", "text": "body"} | fields
    (folder / name).write_text(json.dumps(doc), encoding="utf-8")


def test_loads_documents_sorted_by_file_name(tmp_path: Path) -> None:
    """Order follows file names, so runs are reproducible."""
    write_doc(tmp_path, "b.json", title="Second")
    write_doc(tmp_path, "a.json", title="First")

    docs = load_documents(tmp_path)

    assert [d.title for d in docs] == ["First", "Second"]
    assert docs[0] == Document(title="First", url="https://example.org/t", text="body")


def test_ignores_non_json_files(tmp_path: Path) -> None:
    """Only ``*.json`` files are documents."""
    write_doc(tmp_path, "a.json")
    (tmp_path / "notes.txt").write_text("not a document", encoding="utf-8")

    assert len(load_documents(tmp_path)) == 1


def test_reads_utf8(tmp_path: Path) -> None:
    """Non-ASCII text survives on Windows, whose default encoding is not UTF-8."""
    write_doc(tmp_path, "a.json", text="Résumé – 東京")

    assert load_documents(tmp_path)[0].text == "Résumé – 東京"


def test_missing_folder_raises(tmp_path: Path) -> None:
    """A wrong path fails loudly instead of ingesting nothing."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_documents(tmp_path / "nope")


def test_empty_folder_raises(tmp_path: Path) -> None:
    """An empty corpus is an error, not a silent no-op."""
    with pytest.raises(ValueError, match="No documents"):
        load_documents(tmp_path)


@pytest.mark.parametrize("field", ["title", "text"])
def test_empty_required_field_rejected(tmp_path: Path, field: str) -> None:
    """Blank titles or texts are rejected at load time."""
    write_doc(tmp_path, "a.json", **{field: ""})

    with pytest.raises(ValidationError, match=field):
        load_documents(tmp_path)


def test_documents_are_immutable() -> None:
    """Frozen models cannot be changed after loading."""
    doc = Document(title="T", url="u", text="x")

    with pytest.raises(ValidationError):
        doc.title = "other"  # type: ignore[misc]


def test_committed_corpus_loads() -> None:
    """The 30 sample documents in ``data/raw`` are valid and sectioned."""
    docs = load_documents(DocumindSettings(_env_file=None).data_dir / "raw")

    assert len(docs) == 30
    assert len({d.title for d in docs}) == 30
    assert all("\n== " in d.text for d in docs)
