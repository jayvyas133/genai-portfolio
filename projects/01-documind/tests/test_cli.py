"""Unit tests for the ``documind`` command line.

They cover argument handling, dispatch and the friendly-error paths.
The real ``init-db`` and ``ingest`` against Postgres and Cloudflare are
covered by the integration test.
"""

from pathlib import Path

import pytest
from documind import cli
from documind.config import DocumindSettings

# Port 1 refuses connections immediately; nothing listens there.
UNREACHABLE_DB = "postgresql://u:p@127.0.0.1:1/nope?connect_timeout=2"


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never reach Cloudflare from unit tests, whatever ``.env`` says."""
    monkeypatch.setenv("AICORE_PROVIDER", "fake")


def test_command_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    """No sub-command prints usage and exits with argparse's code 2."""
    with pytest.raises(SystemExit) as exc:
        cli.main([])

    assert exc.value.code == 2
    assert "init-db" in capsys.readouterr().err


def test_init_db_dispatches_and_reports(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``init-db`` passes the configured URL and dimensions to ``init_db``."""
    calls: list[tuple[str, int]] = []

    async def fake_init_db(url: str, dimensions: int) -> bool:
        calls.append((url, dimensions))
        return True

    monkeypatch.setattr(cli, "init_db", fake_init_db)
    monkeypatch.setenv("DOCUMIND_DATABASE_URL", "postgresql://h/db")

    assert cli.main(["init-db"]) == 0
    assert calls == [("postgresql://h/db", DocumindSettings().embedding_dimensions)]
    assert "Database created." in capsys.readouterr().out


def test_unreachable_postgres_is_a_friendly_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A refused connection exits 1 and says how to start Postgres."""
    monkeypatch.setenv("DOCUMIND_DATABASE_URL", UNREACHABLE_DB)

    assert cli.main(["ingest"]) == 1
    assert "docker compose" in caplog.text


def test_missing_corpus_is_a_friendly_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """No ``data/raw`` folder exits 1 before touching the database."""
    monkeypatch.setenv("DOCUMIND_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DOCUMIND_DATABASE_URL", UNREACHABLE_DB)

    assert cli.main(["ingest"]) == 1
    assert "not found" in caplog.text


def test_missing_credentials_is_a_friendly_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Cloudflare without an API token exits 1 naming the variable to set."""
    monkeypatch.setenv("AICORE_PROVIDER", "cloudflare")
    monkeypatch.setenv("AICORE_API_KEY", "")

    assert cli.main(["ingest"]) == 1
    assert "AICORE_API_KEY" in caplog.text


def test_invalid_settings_are_a_friendly_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A bad ``DOCUMIND_*`` value exits 1 with pydantic's message."""
    monkeypatch.setenv("DOCUMIND_CHUNK_OVERLAP_WORDS", "999")

    assert cli.main(["init-db"]) == 1
    assert "chunk_overlap_words must be smaller" in caplog.text


def test_eval_with_missing_golden_set_is_a_friendly_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """No ``evals/golden.jsonl`` exits 1 naming the file, before touching Postgres."""
    monkeypatch.setenv("DOCUMIND_EVALS_DIR", str(tmp_path))
    monkeypatch.setenv("DOCUMIND_DATABASE_URL", UNREACHABLE_DB)

    assert cli.main(["eval"]) == 1
    assert "golden.jsonl: cannot read dataset" in caplog.text


def test_eval_rejects_an_unknown_retriever(capsys: pytest.CaptureFixture[str]) -> None:
    """``--retriever`` accepts only the names ``build_retriever`` knows."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["eval", "--retriever", "bm25"])

    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_eval_rejects_an_unknown_split(capsys: pytest.CaptureFixture[str]) -> None:
    """Only dev, test and all are valid splits."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["eval", "--split", "train"])

    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
