"""Integration tests for ``init_db`` and ``PgChunkStore`` against real Postgres.

They use a separate ``documind_test`` database with 3-dimensional vectors,
so they never touch ingested data. Every test starts from empty tables.
Skipped when Postgres is not reachable
(start it with ``docker compose -f infra/docker-compose.yml up -d postgres``).
"""

from collections.abc import AsyncIterator

import psycopg
import pytest
from documind.chunking import Chunk
from documind.config import DocumindSettings
from documind.corpus import Document
from documind.store import Connection, PgChunkStore, StoreStats, connect, init_db
from psycopg.conninfo import make_conninfo

pytestmark = pytest.mark.integration

DIMS = 3
TEST_URL = make_conninfo(
    DocumindSettings(_env_file=None).database_url, dbname="documind_test", connect_timeout=3
)
DOC = Document(title="CUDA", url="https://example.org/cuda", text="unused")


def chunk(ordinal: int, text: str, section: str = "Intro") -> Chunk:
    """A chunk of the ``CUDA`` test document."""
    return Chunk(document_title="CUDA", ordinal=ordinal, section=section, text=text)


@pytest.fixture
async def conn() -> AsyncIterator[Connection]:
    """A pgvector-ready connection to an empty ``documind_test`` database."""
    try:
        await init_db(TEST_URL, DIMS)
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres not reachable: {exc}")
    async with await connect(TEST_URL) as c:
        await c.execute("TRUNCATE documents, chunks RESTART IDENTITY")
        yield c


async def test_init_db_is_idempotent(conn: Connection) -> None:
    """A second run finds the database and schema already there."""
    assert await init_db(TEST_URL, DIMS) is False


async def test_replace_document_stores_chunks(conn: Connection) -> None:
    """Chunks, sections, hashes and vectors round-trip."""
    store = PgChunkStore(conn)
    chunks = [chunk(0, "alpha"), chunk(1, "beta", section="History")]

    await store.replace_document(DOC, chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    assert await store.stats() == StoreStats(documents=1, chunks=2)
    cur = await conn.execute("SELECT ordinal, section, text, content_hash FROM chunks ORDER BY 1")
    assert await cur.fetchall() == [
        (0, "Intro", "alpha", chunks[0].content_hash),
        (1, "History", "beta", chunks[1].content_hash),
    ]


async def test_replace_swaps_old_chunks_and_updates_document(conn: Connection) -> None:
    """Re-ingesting leaves exactly the new chunks and one document row."""
    store = PgChunkStore(conn)
    await store.replace_document(DOC, [chunk(0, "old"), chunk(1, "old2")], [[1, 0, 0]] * 2)

    moved = DOC.model_copy(update={"url": "https://example.org/new"})
    await store.replace_document(moved, [chunk(0, "new")], [[0, 1, 0]])

    assert await store.stats() == StoreStats(documents=1, chunks=1)
    cur = await conn.execute("SELECT d.url, c.text FROM documents d JOIN chunks c ON true")
    assert await cur.fetchall() == [("https://example.org/new", "new")]


async def test_embeddings_for_returns_only_known_hashes(conn: Connection) -> None:
    """Stored vectors come back by hash; unknown hashes are simply absent."""
    store = PgChunkStore(conn)
    a = chunk(0, "alpha")
    await store.replace_document(DOC, [a], [[0.1, 0.2, 0.3]])

    found = await store.embeddings_for([a.content_hash, "unknown"])

    assert list(found) == [a.content_hash]
    # pgvector stores float32, so compare approximately.
    assert found[a.content_hash] == pytest.approx([0.1, 0.2, 0.3])


async def test_embeddings_for_empty_input(conn: Connection) -> None:
    """No hashes, no query, no results."""
    assert await PgChunkStore(conn).embeddings_for([]) == {}


async def test_length_mismatch_rejected_before_writing(conn: Connection) -> None:
    """Chunks and embeddings must pair up one to one."""
    store = PgChunkStore(conn)

    with pytest.raises(ValueError, match="2 chunks but 1 embeddings"):
        await store.replace_document(DOC, [chunk(0, "a"), chunk(1, "b")], [[1, 0, 0]])

    assert await store.stats() == StoreStats(documents=0, chunks=0)


async def test_failed_replace_keeps_previous_version(conn: Connection) -> None:
    """A bad vector aborts the whole transaction: old chunks survive."""
    store = PgChunkStore(conn)
    await store.replace_document(DOC, [chunk(0, "kept")], [[1, 0, 0]])

    with pytest.raises(psycopg.errors.DataException):
        await store.replace_document(DOC, [chunk(0, "lost")], [[1.0, 0.0]])  # 2 dims

    cur = await conn.execute("SELECT text FROM chunks")
    assert await cur.fetchall() == [("kept",)]


async def test_full_text_column_is_generated(conn: Connection) -> None:
    """``tsv`` is filled automatically and stems words ("warps" ~ "warp")."""
    store = PgChunkStore(conn)
    await store.replace_document(
        DOC, [chunk(0, "Threads are grouped into warps."), chunk(1, "Memory")], [[1, 0, 0]] * 2
    )

    cur = await conn.execute(
        "SELECT ordinal FROM chunks WHERE tsv @@ plainto_tsquery('english', 'warp')"
    )
    assert await cur.fetchall() == [(0,)]
