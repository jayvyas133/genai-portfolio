"""Unit tests for ``ingest``.

``FakeProvider`` embeds offline and an in-memory store replaces Postgres,
so these tests need no network or database.
"""

from collections.abc import Sequence

import pytest
from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.errors import ProviderUnavailableError
from aicore.providers.fakeai import FakeProvider
from aicore.tracing import InMemoryTracer
from documind.chunking import Chunk
from documind.config import DocumindSettings
from documind.corpus import Document
from documind.pipeline import ChunkStore, DimensionMismatchError, IngestReport, ingest

DIMS = 16
SETTINGS = DocumindSettings(
    _env_file=None,
    chunk_words=5,
    chunk_overlap_words=0,
    embed_batch_size=2,
    embedding_dimensions=DIMS,
)


class MemoryStore:
    """In-memory ``ChunkStore``: one entry per document title."""

    def __init__(self) -> None:
        """Start empty."""
        self.docs: dict[str, tuple[Document, list[Chunk], list[list[float]]]] = {}

    async def embeddings_for(self, hashes: Sequence[str]) -> dict[str, list[float]]:
        """Look up vectors by hash across every stored document."""
        known = {
            c.content_hash: v
            for _, chunks, vecs in self.docs.values()
            for c, v in zip(chunks, vecs, strict=True)
        }
        return {h: known[h] for h in hashes if h in known}

    async def replace_document(
        self, doc: Document, chunks: Sequence[Chunk], embeddings: Sequence[list[float]]
    ) -> None:
        """Overwrite the document's entry."""
        self.docs[doc.title] = (doc, list(chunks), list(embeddings))


def doc(title: str, text: str) -> Document:
    """A document with a dummy URL."""
    return Document(title=title, url=f"https://example.org/{title}", text=text)


def client(fake: FakeProvider, tracer: InMemoryTracer | None = None) -> LLMClient:
    """An ``LLMClient`` embedding through ``fake`` with instant retries."""
    settings = LLMSettings(_env_file=None, max_retries=3)  # ignore any local .env
    return LLMClient(fake, settings, base_delay_s=0, tracer=tracer, embedder=fake)


# chunk_words=5 and no headings: 12 words -> 3 chunks (5 + 5 + 2).
GPU = doc("GPU", "a b c d e f g h i j k l")
CPU = doc("CPU", "m n o p q")


def test_memory_store_satisfies_protocol() -> None:
    """mypy checks the assignment; the test documents the intent."""
    store: ChunkStore = MemoryStore()
    assert store is not None


async def test_first_run_embeds_everything_in_batches() -> None:
    """4 chunks with batch size 2 -> 2 embed calls; all stored with vectors."""
    fake, store = FakeProvider(dimensions=DIMS), MemoryStore()

    report = await ingest([GPU, CPU], store, client(fake), SETTINGS)

    assert (report.documents, report.chunks, report.embedded, report.reused) == (2, 4, 4, 0)
    assert report.input_tokens > 0
    # Batches never span documents: GPU needs 2 + 1, CPU needs 1.
    assert [len(r.texts) for r in fake.embed_calls] == [2, 1, 1]
    assert report.embed_calls == 3
    assert all(r.input_type == "document" for r in fake.embed_calls)
    _, chunks, vecs = store.docs["GPU"]
    assert [c.ordinal for c in chunks] == [0, 1, 2]
    assert all(len(v) == DIMS for v in vecs)


async def test_embeds_header_text_not_bare_text() -> None:
    """The model sees ``Title > Section`` context in front of each chunk."""
    fake = FakeProvider(dimensions=DIMS)

    await ingest([CPU], MemoryStore(), client(fake), SETTINGS)

    assert fake.embed_calls[0].texts == ["CPU > Introduction\nm n o p q"]


async def test_rerun_embeds_nothing() -> None:
    """Unchanged documents reuse every stored vector: zero model calls."""
    fake, store = FakeProvider(dimensions=DIMS), MemoryStore()
    await ingest([GPU, CPU], store, client(fake), SETTINGS)
    fake.embed_calls.clear()

    report = await ingest([GPU, CPU], store, client(fake), SETTINGS)

    assert fake.embed_calls == []
    assert (report.embedded, report.reused, report.embed_calls, report.input_tokens) == (0, 4, 0, 0)
    assert report.chunks == 4


async def test_edited_document_embeds_only_changed_chunks() -> None:
    """Changing the last words re-embeds just the chunk that changed."""
    fake, store = FakeProvider(dimensions=DIMS), MemoryStore()
    await ingest([GPU], store, client(fake), SETTINGS)
    fake.embed_calls.clear()

    edited = doc("GPU", "a b c d e f g h i j k z")  # only the third chunk changes

    report = await ingest([edited], store, client(fake), SETTINGS)

    assert (report.embedded, report.reused) == (1, 2)
    assert fake.embed_calls[0].texts == ["GPU > Introduction\nk z"]
    _, chunks, _ = store.docs["GPU"]
    assert chunks[-1].text == "k z"


async def test_duplicate_chunks_in_one_document_embedded_once() -> None:
    """Two identical chunks share a hash: one embedding, stored twice."""
    fake, store = FakeProvider(dimensions=DIMS), MemoryStore()
    twice = doc("Echo", "== A ==\nsame words\n== A ==\nsame words")

    report = await ingest([twice], store, client(fake), SETTINGS)

    assert sum(len(r.texts) for r in fake.embed_calls) == 1
    assert (report.chunks, report.embedded, report.reused) == (2, 1, 1)
    _, _, vecs = store.docs["Echo"]
    assert vecs[0] == vecs[1]


async def test_dimension_mismatch_fails_before_storing() -> None:
    """A wrongly sized model is caught on the first batch; nothing is written."""
    store = MemoryStore()

    with pytest.raises(DimensionMismatchError, match="returned 8 dimensions, expected 16"):
        await ingest([GPU], store, client(FakeProvider(dimensions=8)), SETTINGS)

    assert store.docs == {}


async def test_transient_errors_are_retried_and_traced() -> None:
    """Embedding goes through aicore, so retries and tracing come for free."""
    fake = FakeProvider(dimensions=DIMS, embed_errors=[ProviderUnavailableError("503")])
    tracer = InMemoryTracer()

    report = await ingest([CPU], MemoryStore(), client(fake, tracer), SETTINGS)

    assert report.embedded == 1
    (rec,) = tracer.records
    assert (rec.operation, rec.attempts, rec.outcome) == ("embed", 2, "ok")


async def test_failure_keeps_documents_already_stored() -> None:
    """A crash on document 2 leaves document 1 saved, so a re-run resumes cheaply."""
    fake = FakeProvider(dimensions=DIMS)
    store = MemoryStore()
    await ingest([GPU], store, client(fake), SETTINGS)
    failing = FakeProvider(dimensions=DIMS, embed_errors=[ProviderUnavailableError("503")] * 10)

    with pytest.raises(ProviderUnavailableError):
        await ingest([GPU, CPU], store, client(failing), SETTINGS)

    assert list(store.docs) == ["GPU"]
    # GPU was reused (no calls); CPU failed after 1 + max_retries attempts.
    assert len(failing.embed_calls) == 4
    assert all(r.texts[0].startswith("CPU") for r in failing.embed_calls)


async def test_no_documents() -> None:
    """An empty corpus is a valid no-op."""
    assert await ingest([], MemoryStore(), client(FakeProvider()), SETTINGS) == IngestReport()
