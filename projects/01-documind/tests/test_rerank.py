"""Tests for ``RerankingRetriever`` with the fake reranker (no network, no database).

The fake reranker scores a document by the share of query words it
contains, so the expected order can be worked out by hand.
"""

import pytest
from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.providers.fakeai import FakeProvider
from documind.chunking import Chunk
from documind.retrieval import RerankingRetriever, Retriever
from documind.store import RetrievedChunk


def chunk(cid: int, text: str, title: str = "GPU") -> RetrievedChunk:
    """A retrieved chunk with first-stage score 1.0."""
    return RetrievedChunk(chunk_id=cid, document_title=title, section="S", text=text, score=1.0)


class FixedRetriever:
    """Returns a fixed ranking (cut to k) and records each requested k."""

    def __init__(self, *ranking: RetrievedChunk) -> None:
        """Return ``ranking`` in this order."""
        self.ranking = list(ranking)
        self.requested: list[int] = []

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Record ``k`` and return the first ``k`` chunks."""
        self.requested.append(k)
        return self.ranking[:k]


def reranker(inner: FixedRetriever, fake: FakeProvider, candidates: int = 30) -> RerankingRetriever:
    """A ``RerankingRetriever`` over ``inner`` reranking with ``fake``."""
    client = LLMClient(fake, LLMSettings(_env_file=None), base_delay_s=0, reranker=fake)
    return RerankingRetriever(inner, client, candidates)


POOL = (
    chunk(1, "unrelated text"),
    chunk(2, "memory bandwidth"),
    chunk(3, "HBM memory bandwidth is high"),
)


async def test_reorders_candidates_by_reranker_score() -> None:
    """First-stage order 1, 2, 3; the reranker prefers 3 (all words), then 2."""
    results = await reranker(FixedRetriever(*POOL), FakeProvider()).retrieve(
        "HBM memory bandwidth", 2
    )

    assert [r.chunk_id for r in results] == [3, 2]
    assert [r.score for r in results] == pytest.approx([1.0, 2 / 3])


async def test_fetches_candidates_and_asks_for_top_k() -> None:
    """The inner retriever is asked for ``candidates``; the reranker for top k."""
    inner, fake = FixedRetriever(*POOL), FakeProvider()

    await reranker(inner, fake, candidates=30).retrieve("q", 2)

    assert inner.requested == [30]
    [call] = fake.rerank_calls
    assert call.top_n == 2
    assert len(call.documents) == 3


async def test_asks_for_k_when_k_exceeds_candidates() -> None:
    """k above ``candidates`` raises the depth, so the result still fills."""
    inner = FixedRetriever(*POOL)

    await reranker(inner, FakeProvider(), candidates=2).retrieve("q", 3)

    assert inner.requested == [3]


async def test_sends_chunks_with_their_embedding_header() -> None:
    """The reranker sees ``Title > Section`` + text, exactly as embedded."""
    fake = FakeProvider()

    await reranker(FixedRetriever(chunk(1, "text", title="CUDA")), fake).retrieve("q", 1)

    [call] = fake.rerank_calls
    embedded = Chunk(document_title="CUDA", ordinal=0, section="S", text="text").embed_text
    assert call.documents == [embedded] == ["CUDA > S\ntext"]


async def test_no_candidates_means_no_rerank_call() -> None:
    """An empty first stage returns ``[]`` without paying for a rerank."""
    fake = FakeProvider()

    assert await reranker(FixedRetriever(), fake).retrieve("q", 5) == []
    assert fake.rerank_calls == []


async def test_rejects_bad_arguments() -> None:
    """``candidates`` < 1 fails at construction; ``k`` < 1 on retrieve."""
    with pytest.raises(ValueError, match="candidates must be"):
        reranker(FixedRetriever(), FakeProvider(), candidates=0)
    with pytest.raises(ValueError, match="k must be at least 1"):
        await reranker(FixedRetriever(*POOL), FakeProvider()).retrieve("q", 0)


def test_reranking_retriever_is_a_retriever() -> None:
    """Static check: a reranked retriever can wrap or be wrapped like any other."""
    r: Retriever = reranker(FixedRetriever(), FakeProvider())

    assert r is not None
