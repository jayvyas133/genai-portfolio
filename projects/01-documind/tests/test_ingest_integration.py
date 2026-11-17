"""End-to-end ingestion against real Postgres, with an offline embedder.

``init_db`` -> ``ingest`` -> ``PgChunkStore`` -> pgvector search, all real
except the embedding model (``FakeProvider``, deterministic vectors).
Uses its own ``documind_e2e`` database (16 dimensions), so it never
clashes with ``test_store``'s 3-dimensional ``documind_test``.
Skipped when Postgres is not reachable.
"""

from collections.abc import AsyncIterator

import psycopg
import pytest
from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.models import EmbedRequest
from aicore.providers.fakeai import FakeProvider
from documind.config import DocumindSettings
from documind.corpus import Document
from documind.pipeline import ingest
from documind.store import Connection, PgChunkStore, StoreStats, connect, init_db
from pgvector import Vector
from psycopg.conninfo import make_conninfo

pytestmark = pytest.mark.integration

DIMS = 16
URL = make_conninfo(
    DocumindSettings(_env_file=None).database_url, dbname="documind_e2e", connect_timeout=3
)
SETTINGS = DocumindSettings(
    _env_file=None,
    database_url=URL,
    chunk_words=8,
    chunk_overlap_words=2,
    embed_batch_size=3,
    embedding_dimensions=DIMS,
)
DOCS = [
    Document(
        title="CUDA",
        url="https://example.org/cuda",
        text="CUDA is a parallel computing platform.\n"
        "== History ==\nCUDA was released in 2007 for G80 GPUs by Nvidia.\n"
        "== References ==\nIgnored citation.",
    ),
    Document(
        title="HBM",
        url="https://example.org/hbm",
        text="High Bandwidth Memory stacks DRAM dies vertically.",
    ),
]


@pytest.fixture
async def conn() -> AsyncIterator[Connection]:
    """A connection to an empty ``documind_e2e`` database."""
    try:
        await init_db(URL, DIMS)
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres not reachable: {exc}")
    async with await connect(URL) as c:
        await c.execute("TRUNCATE documents, chunks RESTART IDENTITY")
        yield c


def client(fake: FakeProvider) -> LLMClient:
    """An ``LLMClient`` embedding through ``fake``, ignoring any local ``.env``."""
    return LLMClient(fake, LLMSettings(_env_file=None), base_delay_s=0, embedder=fake)


async def test_ingest_then_rerun_reuses_every_vector(conn: Connection) -> None:
    """First run embeds and stores everything; the second embeds nothing."""
    fake = FakeProvider(dimensions=DIMS)
    store = PgChunkStore(conn)

    first = await ingest(DOCS, store, client(fake), SETTINGS)
    calls_after_first = len(fake.embed_calls)
    second = await ingest(DOCS, store, client(fake), SETTINGS)

    assert first.embedded == first.chunks > 0
    assert (second.embedded, second.reused, second.embed_calls) == (0, first.chunks, 0)
    assert len(fake.embed_calls) == calls_after_first
    assert await store.stats() == StoreStats(documents=2, chunks=first.chunks)


async def test_stored_rows_match_the_chunker(conn: Connection) -> None:
    """Sections, ordinals and boilerplate removal survive the round trip."""
    await ingest(DOCS, PgChunkStore(conn), client(FakeProvider(dimensions=DIMS)), SETTINGS)

    cur = await conn.execute(
        "SELECT d.title, c.ordinal, c.section FROM chunks c"
        " JOIN documents d ON d.id = c.document_id ORDER BY d.title, c.ordinal"
    )
    rows = await cur.fetchall()

    assert rows[0] == ("CUDA", 0, "Introduction")
    assert {section for _, _, section in rows} == {"Introduction", "History"}
    assert [o for t, o, _ in rows if t == "CUDA"] == list(range(sum(r[0] == "CUDA" for r in rows)))


async def test_vector_search_finds_the_matching_chunk(conn: Connection) -> None:
    """Cosine search over the HNSW-indexed column returns the exact chunk first."""
    fake = FakeProvider(dimensions=DIMS)
    await ingest(DOCS, PgChunkStore(conn), client(fake), SETTINGS)
    target = "HBM > Introduction\nHigh Bandwidth Memory stacks DRAM dies vertically."
    query = await client(fake).embed(EmbedRequest(texts=[target], input_type="document"))

    cur = await conn.execute(
        "SELECT c.text, c.embedding <=> %s AS distance FROM chunks c ORDER BY distance LIMIT 1",
        (Vector(query.vectors[0]),),
    )
    row = await cur.fetchone()

    assert row is not None
    assert row[0] == "High Bandwidth Memory stacks DRAM dies vertically."
    assert row[1] == pytest.approx(0, abs=1e-6)
