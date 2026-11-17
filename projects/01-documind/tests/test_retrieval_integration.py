"""Dense retrieval against real Postgres + pgvector, with an offline embedder.

Ingests a small corpus into ``documind_e2e`` (shared with the ingestion
integration test; each test starts from empty tables), then searches it.
Skipped when Postgres is not reachable.
"""

from collections.abc import AsyncIterator

import psycopg
import pytest
from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.providers.fakeai import FakeProvider
from documind.config import DocumindSettings
from documind.corpus import Document
from documind.pipeline import ingest
from documind.retrieval import DenseRetriever
from documind.store import Connection, PgChunkStore, connect, init_db
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
    embedding_dimensions=DIMS,
)
# 50 one-sentence documents: more chunks than HNSW's default 40-row limit.
DOCS = [
    Document(title=f"Doc {i:02d}", url=f"https://example.org/{i}", text=f"Topic {i} facts.")
    for i in range(50)
]


@pytest.fixture
async def store() -> AsyncIterator[PgChunkStore]:
    """A ``PgChunkStore`` over ``documind_e2e`` holding ``DOCS``."""
    try:
        await init_db(URL, DIMS)
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres not reachable: {exc}")
    conn: Connection
    async with await connect(URL) as conn:
        await conn.execute("TRUNCATE documents, chunks RESTART IDENTITY")
        store = PgChunkStore(conn)
        await ingest(DOCS, store, client(FakeProvider(dimensions=DIMS)), SETTINGS)
        yield store


def client(fake: FakeProvider) -> LLMClient:
    """An ``LLMClient`` embedding through ``fake``, ignoring any local ``.env``."""
    return LLMClient(fake, LLMSettings(_env_file=None), base_delay_s=0, embedder=fake)


async def test_exact_text_is_the_top_result(store: PgChunkStore) -> None:
    """Querying a chunk's own embedded text returns that chunk first, score ~1."""
    retriever = DenseRetriever(store, client(FakeProvider(dimensions=DIMS)), DIMS)

    results = await retriever.retrieve("Doc 07 > Introduction\nTopic 7 facts.", 5)

    assert len(results) == 5
    assert (results[0].document_title, results[0].section) == ("Doc 07", "Introduction")
    assert results[0].text == "Topic 7 facts."
    assert results[0].score == pytest.approx(1.0, abs=1e-6)
    # Best first. The index orders by float32 distance while the SELECT
    # returns float64, so near-ties may differ by ~1e-8: allow for that.
    assert all(a.score >= b.score - 1e-6 for a, b in zip(results, results[1:], strict=False))


async def test_k_above_hnsw_default_returns_k_rows(store: PgChunkStore) -> None:
    """k=45 needs ``hnsw.ef_search`` raised; without it HNSW stops at 40 rows."""
    retriever = DenseRetriever(store, client(FakeProvider(dimensions=DIMS)), DIMS)

    results = await retriever.retrieve("Topic 3 facts.", 45)

    assert len(results) == 45
    assert len({r.chunk_id for r in results}) == 45


async def test_k_below_one_is_rejected(store: PgChunkStore) -> None:
    """A zero or negative k is a caller bug, not an empty result."""
    with pytest.raises(ValueError, match="k must be at least 1"):
        await store.search_dense([0.0] * DIMS, 0)
