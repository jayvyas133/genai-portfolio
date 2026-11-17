"""Ingestion: documents -> chunks -> embeddings -> store.

Each document is chunked, then its chunks are matched against the
store by content hash. Only chunks the store has never seen are sent
to the embedding model (through aicore, so calls are traced and
retried). The document is then replaced atomically, so a crash part
way through loses at most the document in progress, and a re-run
re-embeds nothing that was already saved.
"""

import logging
from collections.abc import Sequence
from typing import Protocol

from aicore.client import LLMClient
from aicore.models import EmbedRequest
from pydantic import BaseModel

from documind.chunking import Chunk, chunk_document
from documind.config import DocumindSettings
from documind.corpus import Document

logger = logging.getLogger(__name__)


class ChunkStore(Protocol):
    """What ingestion needs from storage (``PgChunkStore`` satisfies it)."""

    async def embeddings_for(self, hashes: Sequence[str]) -> dict[str, list[float]]:
        """Return stored embeddings for any known ``hashes``."""
        ...

    async def replace_document(
        self,
        doc: Document,
        chunks: Sequence[Chunk],
        embeddings: Sequence[list[float]],
    ) -> None:
        """Store ``doc`` with exactly ``chunks``, atomically."""
        ...


class DimensionMismatchError(Exception):
    """The embedding model returned vectors of an unexpected size."""


class IngestReport(BaseModel):
    """What one ingestion run did.

    ``embedded`` chunks were sent to the model; ``reused`` chunks got
    their vector from the store. ``input_tokens`` is what the model
    reported for the embedded chunks (0 when everything was reused).
    """

    documents: int = 0
    chunks: int = 0
    embedded: int = 0
    reused: int = 0
    embed_calls: int = 0
    input_tokens: int = 0


async def ingest(
    documents: Sequence[Document],
    store: ChunkStore,
    client: LLMClient,
    settings: DocumindSettings,
) -> IngestReport:
    """Chunk, embed (only what is new) and store every document.

    Raises:
        DimensionMismatchError: If the model's vectors do not have
            ``settings.embedding_dimensions`` dimensions.
        aicore.errors.LLMError: If embedding fails after retries.
    """
    report = IngestReport()
    for doc in documents:
        chunks = chunk_document(doc, settings.chunk_words, settings.chunk_overlap_words)
        vectors = await store.embeddings_for([c.content_hash for c in chunks])
        # Unique by hash: identical chunks in one document are embedded once.
        unseen = {c.content_hash: c for c in chunks if c.content_hash not in vectors}
        missing = list(unseen.values())

        for start in range(0, len(missing), settings.embed_batch_size):
            batch = missing[start : start + settings.embed_batch_size]
            vectors.update(await _embed_batch(batch, client, settings, report))

        await store.replace_document(doc, chunks, [vectors[c.content_hash] for c in chunks])
        report.documents += 1
        report.chunks += len(chunks)
        report.embedded += len(missing)
        report.reused += len(chunks) - len(missing)
        logger.info("Stored %r: %d chunks, %d embedded", doc.title, len(chunks), len(missing))
    return report


async def _embed_batch(
    batch: Sequence[Chunk],
    client: LLMClient,
    settings: DocumindSettings,
    report: IngestReport,
) -> dict[str, list[float]]:
    """Embed one batch of chunks; return ``{content_hash: vector}``.

    Chunks are embedded with their ``Title > Section`` header, as
    documents (no query instruction). Updates ``report``'s call and
    token counts.

    Raises:
        DimensionMismatchError: If vectors have the wrong size.
    """
    response = await client.embed(
        EmbedRequest(texts=[c.embed_text for c in batch], input_type="document")
    )
    if response.dimensions != settings.embedding_dimensions:
        msg = (
            f"Embedding model returned {response.dimensions} dimensions, "
            f"expected {settings.embedding_dimensions} (DOCUMIND_EMBEDDING_DIMENSIONS)"
        )
        raise DimensionMismatchError(msg)
    report.embed_calls += 1
    report.input_tokens += response.usage.input_tokens
    return dict(zip((c.content_hash for c in batch), response.vectors, strict=True))
