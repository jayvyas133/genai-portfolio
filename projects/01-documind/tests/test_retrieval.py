"""Unit tests for ``DenseRetriever`` with a stub index (no database)."""

from collections.abc import Sequence

import pytest
from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.providers.fakeai import FakeProvider
from documind.pipeline import DimensionMismatchError
from documind.retrieval import DenseRetriever, Retriever
from documind.store import RetrievedChunk

DIMS = 8
RESULT = RetrievedChunk(chunk_id=1, document_title="CUDA", section="History", text="t", score=0.9)


class StubIndex:
    """Records each search and returns one fixed result."""

    def __init__(self) -> None:
        """Start with no recorded searches."""
        self.searches: list[tuple[list[float], int]] = []

    async def search_dense(self, vector: Sequence[float], k: int) -> list[RetrievedChunk]:
        """Record the call and return ``[RESULT]``."""
        self.searches.append((list(vector), k))
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


def test_dense_retriever_is_a_retriever() -> None:
    """Static check that ``DenseRetriever`` satisfies the ``Retriever`` protocol."""
    r: Retriever = retriever(StubIndex(), FakeProvider(dimensions=DIMS))

    assert r is not None
