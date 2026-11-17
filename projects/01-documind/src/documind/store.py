"""Postgres + pgvector storage for documents and their chunks.

Schema:
    documents  one row per source document (unique title).
    chunks     text, section path, content hash and embedding per chunk,
               plus a generated ``tsv`` column for full-text search.

Ingestion replaces a document's chunks inside one transaction, so a
reader never sees half a document. Embeddings are looked up by content
hash first, so unchanged chunks are never sent to the embedding model.
"""

from collections.abc import Sequence
from typing import Any

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector_async
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from pydantic import BaseModel, ConfigDict

from documind.chunking import Chunk
from documind.corpus import Document

Connection = psycopg.AsyncConnection[tuple[Any, ...]]

# {dims} is filled in with psycopg.sql, never with string formatting.
_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id    BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title TEXT NOT NULL UNIQUE,
    url   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id  BIGINT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    ordinal      INT    NOT NULL,
    section      TEXT   NOT NULL,
    text         TEXT   NOT NULL,
    content_hash TEXT   NOT NULL,
    embedding    vector({dims}) NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS
                 (to_tsvector('english', section || ' ' || text)) STORED,
    UNIQUE (document_id, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_content_hash ON chunks (content_hash);
"""


# pgvector's HNSW search returns at most ``hnsw.ef_search`` rows (default 40).
_DEFAULT_EF_SEARCH = 40


class StoreStats(BaseModel):
    """Row counts, for CLI output and the README."""

    documents: int
    chunks: int


class RetrievedChunk(BaseModel):
    """One search result, with where it came from and how well it matched.

    ``score`` is higher-is-better; for dense search it is cosine
    similarity (1 - cosine distance), between -1 and 1.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: int
    document_title: str
    section: str
    text: str
    score: float


def _database_name(database_url: str) -> str:
    """Return the database name in ``database_url``.

    Raises:
        ValueError: If the URL names no database.
    """
    name = conninfo_to_dict(database_url).get("dbname")
    if not name:
        msg = f"DATABASE_URL must name a database: {database_url!r}"
        raise ValueError(msg)
    return str(name)


async def init_db(database_url: str, dimensions: int) -> bool:
    """Create the database (if missing), the extension and the schema.

    Safe to run any number of times. Connects to the server's built-in
    ``postgres`` database first, because a database cannot create itself.

    Args:
        database_url: Target database, e.g. ``postgresql://.../documind``.
        dimensions: Embedding size; fixed into the ``vector(n)`` column.

    Returns:
        True if the database was created by this call.
    """
    name = _database_name(database_url)
    admin_url = make_conninfo(database_url, dbname="postgres")
    async with await psycopg.AsyncConnection.connect(admin_url, autocommit=True) as admin:
        cur = await admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))
        created = await cur.fetchone() is None
        if created:
            # CREATE DATABASE has no IF NOT EXISTS and cannot run in a transaction.
            await admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))

    schema = sql.SQL(_SCHEMA).format(dims=sql.Literal(dimensions))
    async with await psycopg.AsyncConnection.connect(database_url, autocommit=True) as conn:
        await conn.execute(schema)
    return created


async def connect(database_url: str) -> Connection:
    """Open a connection that can send and receive pgvector values.

    Run ``init_db`` first: registration needs the ``vector`` type to exist.
    The caller closes the connection (``async with await connect(...)``).
    """
    conn = await psycopg.AsyncConnection.connect(database_url, autocommit=True)
    await register_vector_async(conn)
    return conn


class PgChunkStore:
    """Reads and writes chunks through one open connection."""

    def __init__(self, conn: Connection) -> None:
        """Use ``conn``, which must come from ``connect``."""
        self._conn = conn

    async def embeddings_for(self, hashes: Sequence[str]) -> dict[str, list[float]]:
        """Return stored embeddings for any of ``hashes`` already in the DB.

        Lookup is across all documents: identical content is embedded once.
        """
        if not hashes:
            return {}
        cur = await self._conn.execute(
            "SELECT DISTINCT ON (content_hash) content_hash, embedding"
            " FROM chunks WHERE content_hash = ANY(%s)",
            (list(hashes),),
        )
        return {h: vec.to_list() for h, vec in await cur.fetchall()}

    async def replace_document(
        self,
        doc: Document,
        chunks: Sequence[Chunk],
        embeddings: Sequence[list[float]],
    ) -> None:
        """Store ``doc`` with exactly ``chunks``, atomically.

        Upserts the document row, deletes its old chunks and inserts the
        new ones in one transaction. ``embeddings[i]`` belongs to
        ``chunks[i]``.

        Raises:
            ValueError: If the two sequences differ in length.
        """
        if len(chunks) != len(embeddings):
            msg = f"{len(chunks)} chunks but {len(embeddings)} embeddings"
            raise ValueError(msg)
        async with self._conn.transaction():
            cur = await self._conn.execute(
                "INSERT INTO documents (title, url) VALUES (%s, %s)"
                " ON CONFLICT (title) DO UPDATE SET url = EXCLUDED.url"
                " RETURNING id",
                (doc.title, doc.url),
            )
            row = await cur.fetchone()
            assert row is not None  # RETURNING always yields the row
            doc_id = row[0]
            await self._conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
            async with self._conn.cursor() as insert:
                await insert.executemany(
                    "INSERT INTO chunks"
                    " (document_id, ordinal, section, text, content_hash, embedding)"
                    " VALUES (%s, %s, %s, %s, %s, %s)",
                    [
                        (doc_id, c.ordinal, c.section, c.text, c.content_hash, Vector(e))
                        for c, e in zip(chunks, embeddings, strict=True)
                    ],
                )

    async def search_dense(self, vector: Sequence[float], k: int) -> list[RetrievedChunk]:
        """Return the ``k`` chunks closest to ``vector`` by cosine distance.

        Uses the HNSW (Hierarchical Navigable Small World) index, which is
        approximate. HNSW returns at most ``hnsw.ef_search`` rows, so for
        ``k`` above the default 40 it is raised for this query only
        (``SET LOCAL`` inside a transaction).

        Raises:
            ValueError: If ``k`` < 1.
        """
        if k < 1:
            msg = f"k must be at least 1, got {k}"
            raise ValueError(msg)
        async with self._conn.transaction():
            if k > _DEFAULT_EF_SEARCH:
                await self._conn.execute(
                    sql.SQL("SET LOCAL hnsw.ef_search = {}").format(sql.Literal(k))
                )
            cur = await self._conn.execute(
                "SELECT c.id, d.title, c.section, c.text, c.embedding <=> %s AS distance"
                " FROM chunks c JOIN documents d ON d.id = c.document_id"
                " ORDER BY distance LIMIT %s",
                (Vector(list(vector)), k),
            )
            rows = await cur.fetchall()
        return [
            RetrievedChunk(
                chunk_id=cid, document_title=title, section=section, text=text, score=1 - dist
            )
            for cid, title, section, text, dist in rows
        ]

    async def stats(self) -> StoreStats:
        """Count documents and chunks."""
        cur = await self._conn.execute(
            "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks)"
        )
        row = await cur.fetchone()
        assert row is not None
        return StoreStats(documents=row[0], chunks=row[1])
