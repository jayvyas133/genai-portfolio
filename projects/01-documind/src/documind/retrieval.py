"""Retrievers: turn a question into a ranked list of chunks.

All retrievers share one interface (``Retriever``), so callers can swap
them freely.
"""

from collections.abc import Sequence
from typing import Literal, Protocol, assert_never, get_args

from aicore.client import LLMClient
from aicore.models import EmbedRequest, RerankRequest

from documind.pipeline import DimensionMismatchError
from documind.store import RetrievedChunk

# Standard RRF constant (Cormack et al., 2009). It keeps top ranks close
# (1/61 vs 1/62), so agreement between lists beats one list's first place.
RRF_C = 60

RetrieverName = Literal["dense", "sparse", "hybrid", "dense-rerank", "hybrid-rerank"]
RETRIEVER_NAMES: tuple[RetrieverName, ...] = get_args(RetrieverName)


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


class SparseIndex(Protocol):
    """A store that can search by keywords, such as ``PgChunkStore``."""

    async def search_sparse(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return up to ``k`` chunks matching words of ``query``, best first."""
        ...


class ChunkIndex(DenseIndex, SparseIndex, Protocol):
    """A store with both dense and sparse search, such as ``PgChunkStore``."""


def build_retriever(
    name: RetrieverName, store: ChunkIndex, client: LLMClient, dimensions: int
) -> Retriever:
    """Assemble the retriever called ``name`` over ``store``.

    ``-rerank`` names add a reranker on top of the first stage.
    """
    dense = DenseRetriever(store, client, dimensions)
    match name:
        case "dense":
            return dense
        case "sparse":
            return SparseRetriever(store)
        case "hybrid":
            return HybridRetriever(dense, SparseRetriever(store))
        case "dense-rerank":
            return RerankingRetriever(dense, client)
        case "hybrid-rerank":
            return RerankingRetriever(HybridRetriever(dense, SparseRetriever(store)), client)
        case _:
            assert_never(name)


class DenseRetriever:
    """Embed the query, then return the nearest chunks by cosine distance.

    ``input_type="query"`` lets aicore add the query prefix BGE (BAAI
    General Embedding) expects.
    """

    def __init__(self, store: DenseIndex, client: LLMClient, dimensions: int) -> None:
        """Search ``store``, embedding queries with ``client``.

        ``dimensions`` is the store's vector size, used to catch a changed
        embedding model.
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


class SparseRetriever:
    """Keyword search with Postgres full-text.

    Good at exact names and numbers ("RTX 3090"), bad at paraphrases.
    No model call, so fast and free.
    """

    def __init__(self, store: SparseIndex) -> None:
        """Search ``store``."""
        self._store = store

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return up to ``k`` chunks matching words of ``query``, best first.

        May return fewer than ``k``, or none.

        Raises:
            ValueError: If ``k`` < 1.
        """
        return await self._store.search_sparse(query, k)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]], c: int = RRF_C
) -> list[RetrievedChunk]:
    """Merge ranked lists into one: each chunk scores the sum of 1 / (c + rank).

    Uses ranks only, because the lists' scores are on different scales.
    Returned chunks carry the fused score; ties keep first-seen order.

    Raises:
        ValueError: If ``c`` is negative.
    """
    if c < 0:
        msg = f"c must be at least 0, got {c}"
        raise ValueError(msg)
    scores: dict[int, float] = {}
    chunks: dict[int, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1 / (c + rank)
            chunks.setdefault(chunk.chunk_id, chunk)
    # Stable sort over an insertion-ordered dict keeps ties first-seen.
    order = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [chunks[cid].model_copy(update={"score": scores[cid]}) for cid in order]


class HybridRetriever:
    """Dense and sparse search, merged with RRF (Reciprocal Rank Fusion).

    Dense catches paraphrases, sparse catches exact terms. Searches run
    sequentially because they share one Postgres connection.
    """

    def __init__(
        self, dense: Retriever, sparse: Retriever, candidates: int = 50, c: int = RRF_C
    ) -> None:
        """Fuse ``dense`` and ``sparse``, each asked for ``candidates`` chunks.

        Raises:
            ValueError: If ``candidates`` < 1 or ``c`` < 0.
        """
        if candidates < 1:
            msg = f"candidates must be at least 1, got {candidates}"
            raise ValueError(msg)
        if c < 0:
            msg = f"c must be at least 0, got {c}"
            raise ValueError(msg)
        self._dense = dense
        self._sparse = sparse
        self._candidates = candidates
        self._c = c

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return the top ``k`` chunks of the fused dense and sparse results.

        Raises:
            ValueError: If ``k`` < 1.
            Any error from the underlying retrievers.
        """
        if k < 1:
            msg = f"k must be at least 1, got {k}"
            raise ValueError(msg)
        depth = max(k, self._candidates)
        dense = await self._dense.retrieve(query, depth)
        sparse = await self._sparse.retrieve(query, depth)
        return reciprocal_rank_fusion([dense, sparse], self._c)[:k]


class RerankingRetriever:
    """Re-sort another retriever's top candidates with a cross-encoder.

    The reranker reads question and chunk together: slower than embeddings
    but more precise, so it only sees the top ``candidates``. Chunks are
    sent with their ``Title > Section`` header for context.
    """

    def __init__(self, inner: Retriever, client: LLMClient, candidates: int = 30) -> None:
        """Rerank the top ``candidates`` chunks from ``inner`` with ``client``.

        Raises:
            ValueError: If ``candidates`` < 1.
        """
        if candidates < 1:
            msg = f"candidates must be at least 1, got {candidates}"
            raise ValueError(msg)
        self._inner = inner
        self._client = client
        self._candidates = candidates

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return the ``k`` candidates the reranker scores highest, best first.

        Returned chunks carry the reranker's score.

        Raises:
            ValueError: If ``k`` < 1.
            aicore.errors.LLMError: If the rerank call fails after retries.
            Any error from ``inner``.
        """
        if k < 1:
            msg = f"k must be at least 1, got {k}"
            raise ValueError(msg)
        pool = await self._inner.retrieve(query, max(k, self._candidates))
        if not pool:
            return []
        response = await self._client.rerank(
            RerankRequest(query=query, documents=[c.embed_text for c in pool], top_n=k)
        )
        return [pool[r.index].model_copy(update={"score": r.score}) for r in response.results]
