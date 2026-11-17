"""Retrievers: turn a question into a ranked list of chunks.

Every retriever has the same shape (``Retriever``), so the eval runner
and, later, the API can swap dense, hybrid and reranked retrieval without
changing. M3 has the dense baseline; M4 adds the others and measures
them against it.
"""

from collections.abc import Sequence
from typing import Protocol

from aicore.client import LLMClient
from aicore.models import EmbedRequest

from documind.pipeline import DimensionMismatchError
from documind.store import RetrievedChunk


class Retriever(Protocol):
    """Anything that returns the ``k`` best chunks for a query, best first."""

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return up to ``k`` chunks ranked by relevance to ``query``."""
        ...


class DenseIndex(Protocol):
    """A store that can search by vector, such as ``PgChunkStore``."""

    async def search_dense(self, vector: Sequence[float], k: int) -> list[RetrievedChunk]:
        """Return the ``k`` chunks nearest to ``vector``, best first."""
        ...


class DenseRetriever:
    """Embed the query, then take the nearest chunks by cosine distance.

    The query is embedded with ``input_type="query"``: BGE (BAAI General
    Embedding) expects an instruction prefix on short queries but not on
    passages, and aicore's adapter adds it. Chunks were embedded as
    documents during ingestion.
    """

    def __init__(self, store: DenseIndex, client: LLMClient, dimensions: int) -> None:
        """Search ``store``, embedding queries with ``client``.

        ``dimensions`` is the vector size the store was built with; a
        query vector of another size means the embedding model changed.
        """
        self._store = store
        self._client = client
        self._dimensions = dimensions

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return the ``k`` chunks whose embeddings are closest to ``query``'s.

        Raises:
            DimensionMismatchError: If the query vector does not match the store.
            ValueError: If ``k`` < 1.
            aicore.errors.LLMError: If the embedding call fails after retries.
        """
        response = await self._client.embed(EmbedRequest(texts=[query], input_type="query"))
        vector = response.vectors[0]
        if len(vector) != self._dimensions:
            msg = (
                f"query vector has {len(vector)} dimensions, store expects "
                f"{self._dimensions}; was the embedding model changed?"
            )
            raise DimensionMismatchError(msg)
        return await self._store.search_dense(vector, k)
