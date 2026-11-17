"""Tests for ``DocumindSettings``.

Each test clears ``DOCUMIND_*`` variables and ignores any ``.env`` file, so
results do not depend on the developer's machine.
"""

import os
from pathlib import Path

import pytest
from documind.config import DocumindSettings
from pydantic import ValidationError


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every ``DOCUMIND_*`` variable for the duration of a test."""
    for key in list(os.environ):
        if key.startswith("DOCUMIND_"):
            monkeypatch.delenv(key)


def load() -> DocumindSettings:
    """Build settings from the environment only, never from ``.env``."""
    return DocumindSettings(_env_file=None)


def test_defaults() -> None:
    """Defaults match the plan: local documind DB, 300/50 words, 768 dims."""
    s = load()

    assert s.database_url.endswith("@localhost:5432/documind")
    assert (s.chunk_words, s.chunk_overlap_words) == (300, 50)
    assert s.embed_batch_size == 50
    assert s.embedding_dimensions == 768


def test_data_dir_is_inside_the_project() -> None:
    """``data_dir`` resolves to ``projects/01-documind/data`` regardless of cwd."""
    s = load()

    assert s.data_dir.is_absolute()
    assert s.data_dir.name == "data"
    assert (s.data_dir.parent / "pyproject.toml").is_file()


def test_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``DOCUMIND_*`` variables override defaults, with type conversion."""
    monkeypatch.setenv("DOCUMIND_DATABASE_URL", "postgresql://u:p@db:5432/x")
    monkeypatch.setenv("DOCUMIND_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DOCUMIND_CHUNK_WORDS", "200")
    monkeypatch.setenv("DOCUMIND_EMBED_BATCH_SIZE", "10")

    s = load()

    assert s.database_url == "postgresql://u:p@db:5432/x"
    assert s.data_dir == tmp_path
    assert s.chunk_words == 200
    assert s.embed_batch_size == 10


def test_aicore_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """A shared ``.env`` holding ``AICORE_*`` values does not break loading."""
    monkeypatch.setenv("AICORE_PROVIDER", "cloudflare")

    assert load().chunk_words == 300


@pytest.mark.parametrize("overlap", [300, 301])
def test_overlap_must_be_smaller_than_chunk(overlap: int) -> None:
    """An overlap >= chunk size would stop the chunker advancing."""
    with pytest.raises(ValidationError, match="chunk_overlap_words"):
        DocumindSettings(_env_file=None, chunk_words=300, chunk_overlap_words=overlap)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chunk_words", 0),
        ("chunk_overlap_words", -1),
        ("embed_batch_size", 0),
        ("embed_batch_size", 101),
        ("embedding_dimensions", 0),
    ],
)
def test_out_of_range_values_rejected(field: str, value: int) -> None:
    """Bounds guard against typos (101 would exceed Cloudflare's batch cap)."""
    with pytest.raises(ValidationError, match=field):
        DocumindSettings.model_validate({field: value})
