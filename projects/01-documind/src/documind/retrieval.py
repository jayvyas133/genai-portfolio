"""Retrievers: turn a question into a ranked list of chunks.

Every retriever has the same shape (``Retriever``), so the eval runner
and, later, the API can swap dense, hybrid and reranked retrieval without
changing. M3 has the dense baseline; M4 adds sparse (full-text),
hybrid and reranked retrieval and measures them against it.
"""

from collections.abc import Sequence
from typing import Literal, Protocol, assert_never, get_args

from aicore.client import LLMClient
from aicore.models import EmbedRequest, RerankRequest

from documind.pipeline import DimensionMismatchError
from documind.store import RetrievedChunk

# The constant from the original RRF paper (Cormack et al., 2009). It damps
# the gap between top ranks: with c = 60, rank 1 scores 1/61 and rank 2
# 1/62, so one list's first place cannot outvote agreement between lists.
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

    ``-rerank`` names wrap the first stage in ``RerankingRetriever``, so
    ``dense-rerank`` and ``hybrid-rerank`` show whether the reranker
    still needs hybrid search underneath it. Default depths are used:
    50 candidates per search for hybrid, 30 for the reranker.
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


class SparseRetriever:
    """Keyword search: rank chunks by the question's words (Postgres full-text).

    Finds exact names and numbers ("RTX 3090", "HBM2e") that an embedding
    can blur, but misses paraphrases. No model call, so it is fast and free.
    """

    def __init__(self, store: SparseIndex) -> None:
        """Search ``store``."""
        self._store = store

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return up to ``k`` chunks matching words of ``query``, best first.

        May return fewer than ``k`` (or none) when few chunks share a word
        with the query.

        Raises:
            ValueError: If ``k`` < 1.
        """
        return await self._store.search_sparse(query, k)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]], c: int = RRF_C
) -> list[RetrievedChunk]:
    """Merge ranked lists into one: each chunk scores the sum of 1 / (c + rank).

    Only ranks are used, never the lists' own scores: cosine similarity
    and ``ts_rank_cd`` are on different scales, so adding them would let
    one list dominate. A chunk near the top of several lists beats one at
    the top of a single list. Returned chunks carry the fused score.
    Ties keep the order in which chunks were first seen (earlier lists
    first), so results are repeatable. Each list is assumed to hold a
    chunk at most once.

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
    # sorted() is stable and dicts keep insertion order: ties stay first-seen.
    order = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [chunks[cid].model_copy(update={"score": scores[cid]}) for cid in order]


class HybridRetriever:
    """Dense and sparse search, merged with RRF (Reciprocal Rank Fusion).

    Dense search finds paraphrases; sparse search finds exact names and
    numbers. Each fetches ``candidates`` chunks, so a chunk ranked 30th
    by one search can still rise if the other ranks it well; the fused
    list is cut to ``k``.

    The two searches run one after the other, not concurrently: they share
    one Postgres connection, which runs one query at a time, and sparse
    search takes milliseconds next to the embedding call.
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

        Each search is asked for ``max(k, candidates)`` chunks, so ``k``
        larger than ``candidates`` still fills.

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
    """Re-sort another retriever's top candidates with a cross-encoder reranker.

    The embedder scores the question and each chunk separately, then
    compares vectors. A reranker (``bge-reranker-base``) reads the
    question and one chunk together and scores how well that chunk
    answers it: slower, but more precise. So it only sees ``candidates``
    chunks, and the first-stage retriever's job becomes recall.

    Each chunk is sent with its ``Title > Section`` header, as it was
    embedded, so the reranker knows which document "it" refers to.
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

        Returned chunks carry the reranker's score. No rerank call is made
        when ``inner`` finds nothing.

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
