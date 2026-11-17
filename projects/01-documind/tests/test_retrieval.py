"""Unit tests for the retrievers with a stub index (no database)."""

from collections.abc import Sequence

import pytest
from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.providers.fakeai import FakeProvider
from documind.pipeline import DimensionMismatchError
from documind.retrieval import (
    RETRIEVER_NAMES,
    DenseRetriever,
    Retriever,
    RetrieverName,
    SparseRetriever,
    build_retriever,
)
from documind.store import RetrievedChunk

DIMS = 8
RESULT = RetrievedChunk(chunk_id=1, document_title="CUDA", section="History", text="t", score=0.9)


class StubIndex:
    """Records each search and returns one fixed result."""

    def __init__(self) -> None:
        """Start with no recorded searches."""
        self.searches: list[tuple[list[float], int]] = []
        self.keyword_searches: list[tuple[str, int]] = []

    async def search_dense(self, vector: Sequence[float], k: int) -> list[RetrievedChunk]:
        """Record the call and return ``[RESULT]``."""
        self.searches.append((list(vector), k))
        return [RESULT]

    async def search_sparse(self, query: str, k: int) -> list[RetrievedChunk]:
        """Record the call and return ``[RESULT]``."""
        self.keyword_searches.append((query, k))
        return [RESULT]


def retriever(index: StubIndex, fake: FakeProvider, dims: int = DIMS) -> DenseRetriever:
    """A ``DenseRetriever`` over ``index`` embedding with ``fake``."""
    client = LLMClient(fake, LLMSettings(_env_file=None), base_delay_s=0, embedder=fake)
    return DenseRetriever(index, client, dims)


async def test_embeds_the_query_as_a_query_and_searches() -> None:
    """One embed call with ``input_type="query"``; its vector and k reach the index."""
    fake, index = FakeProvider(dimensions=DIMS), StubIndex()

    results = await retriever(index, fake).retrieve("When was CUDA released?", 5)

    assert results == [RESULT]
    [call] = fake.embed_calls
    assert (call.texts, call.input_type) == (["When was CUDA released?"], "query")
    [(vector, k)] = index.searches
    assert (len(vector), k) == (DIMS, 5)


async def test_wrong_vector_size_fails_before_searching() -> None:
    """A model swap (different dimensions) is caught, not sent to Postgres."""
    index = StubIndex()

    with pytest.raises(DimensionMismatchError, match="8 dimensions, store expects 768"):
        await retriever(index, FakeProvider(dimensions=DIMS), dims=768).retrieve("q", 5)
    assert index.searches == []


async def test_sparse_passes_the_raw_question_to_the_index() -> None:
    """No model call: the question text and k go straight to full-text search."""
    index = StubIndex()

    results = await SparseRetriever(index).retrieve("When was CUDA released?", 5)

    assert results == [RESULT]
    assert index.keyword_searches == [("When was CUDA released?", 5)]


@pytest.mark.parametrize(
    ("name", "embeds", "keyword_searches", "reranks"),
    [
        ("dense", 1, 0, 0),
        ("sparse", 0, 1, 0),
        ("hybrid", 1, 1, 0),
        ("dense-rerank", 1, 0, 1),
        ("hybrid-rerank", 1, 1, 1),
    ],
)
async def test_build_retriever_wires_each_name(
    name: RetrieverName, embeds: int, keyword_searches: int, reranks: int
) -> None:
    """Each name calls exactly the searches and models it promises."""
    fake, index = FakeProvider(dimensions=DIMS), StubIndex()
    client = LLMClient(
        fake, LLMSettings(_env_file=None), base_delay_s=0, embedder=fake, reranker=fake
    )

    results = await build_retriever(name, index, client, DIMS).retrieve("CUDA", 1)

    assert [r.chunk_id for r in results] == [RESULT.chunk_id]
    assert len(fake.embed_calls) == len(index.searches) == embeds
    assert len(index.keyword_searches) == keyword_searches
    assert len(fake.rerank_calls) == reranks


def test_every_retriever_name_is_covered() -> None:
    """A new name added to ``RetrieverName`` must be added to the test above."""
    assert RETRIEVER_NAMES == ("dense", "sparse", "hybrid", "dense-rerank", "hybrid-rerank")


def test_retrievers_satisfy_the_protocol() -> None:
    """Static check that every retriever satisfies ``Retriever``."""
    retrievers: list[Retriever] = [
        retriever(StubIndex(), FakeProvider(dimensions=DIMS)),
        SparseRetriever(StubIndex()),
    ]

    assert len(retrievers) == 2
