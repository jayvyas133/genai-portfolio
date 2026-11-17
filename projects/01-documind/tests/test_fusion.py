"""Tests for RRF (Reciprocal Rank Fusion) and the hybrid retriever, no database."""

import pytest
from documind.retrieval import HybridRetriever, Retriever, reciprocal_rank_fusion
from documind.store import RetrievedChunk


def chunk(cid: int, score: float = 0.0) -> RetrievedChunk:
    """A chunk identified by ``cid``; other fields do not matter here."""
    return RetrievedChunk(
        chunk_id=cid, document_title=f"Doc {cid}", section="S", text="t", score=score
    )


def ids(chunks: list[RetrievedChunk]) -> list[int]:
    """The chunk ids, in order."""
    return [c.chunk_id for c in chunks]


class FixedRetriever:
    """Returns a fixed ranking (cut to k) and records each requested k."""

    def __init__(self, *chunk_ids: int) -> None:
        """Rank chunks ``chunk_ids`` in this order."""
        self.ranking = [chunk(cid, score=100.0 - i) for i, cid in enumerate(chunk_ids)]
        self.requested: list[int] = []

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Record ``k`` and return the first ``k`` chunks."""
        self.requested.append(k)
        return self.ranking[:k]


def test_rrf_matches_hand_computed_scores() -> None:
    """Lists [1, 2, 3] and [3, 1, 4] with c = 60.

    1: 1/61 + 1/62    3: 1/63 + 1/61    2: 1/62    4: 1/63
    Chunk 1 wins: both lists rank it high. Chunk 3 is first in one list
    but third in the other, so it comes second.
    """
    fused = reciprocal_rank_fusion([[chunk(1), chunk(2), chunk(3)], [chunk(3), chunk(1), chunk(4)]])

    assert ids(fused) == [1, 3, 2, 4]
    assert [c.score for c in fused] == pytest.approx(
        [1 / 61 + 1 / 62, 1 / 63 + 1 / 61, 1 / 62, 1 / 63]
    )


def test_rrf_ignores_the_input_scores() -> None:
    """A huge score in one list does not matter; only rank does."""
    fused = reciprocal_rank_fusion([[chunk(1, score=0.1)], [chunk(2, score=999.0)]])

    assert fused[0].score == fused[1].score == pytest.approx(1 / 61)


def test_rrf_ties_keep_first_seen_order() -> None:
    """Equal fused scores: the chunk from the earlier list comes first."""
    assert ids(reciprocal_rank_fusion([[chunk(1)], [chunk(2)]])) == [1, 2]
    assert ids(reciprocal_rank_fusion([[chunk(2)], [chunk(1)]])) == [2, 1]


def test_rrf_of_nothing_is_empty() -> None:
    """No lists, or only empty lists, fuse to an empty list."""
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_rejects_negative_c() -> None:
    """A negative c could divide by zero at rank -c."""
    with pytest.raises(ValueError, match="c must be at least 0"):
        reciprocal_rank_fusion([[chunk(1)]], c=-1)


async def test_hybrid_fetches_candidates_from_both_and_cuts_to_k() -> None:
    """Both searches are asked for ``candidates`` chunks; the fused list is cut to k."""
    dense, sparse = FixedRetriever(1, 2, 3), FixedRetriever(3, 1, 4)

    results = await HybridRetriever(dense, sparse, candidates=50).retrieve("q", 2)

    assert ids(results) == [1, 3]
    assert (dense.requested, sparse.requested) == ([50], [50])


async def test_hybrid_asks_for_k_when_k_exceeds_candidates() -> None:
    """k above ``candidates`` raises the depth, so the result still fills."""
    dense, sparse = FixedRetriever(1), FixedRetriever(2)

    await HybridRetriever(dense, sparse, candidates=5).retrieve("q", 10)

    assert (dense.requested, sparse.requested) == ([10], [10])


async def test_hybrid_falls_back_to_dense_when_sparse_finds_nothing() -> None:
    """A question sharing no word with the corpus still gets dense results."""
    results = await HybridRetriever(FixedRetriever(1, 2), FixedRetriever()).retrieve("q", 5)

    assert ids(results) == [1, 2]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"candidates": 0}, "candidates must be"), ({"c": -1}, "c must be")],
)
def test_hybrid_rejects_bad_settings(kwargs: dict[str, int], match: str) -> None:
    """Bad settings fail at construction, not on the first query."""
    with pytest.raises(ValueError, match=match):
        HybridRetriever(FixedRetriever(), FixedRetriever(), **kwargs)


async def test_hybrid_rejects_k_below_one() -> None:
    """Same contract as the other retrievers."""
    with pytest.raises(ValueError, match="k must be at least 1"):
        await HybridRetriever(FixedRetriever(), FixedRetriever()).retrieve("q", 0)


def test_hybrid_is_a_retriever() -> None:
    """Static check: hybrid can be used anywhere a ``Retriever`` is expected."""
    r: Retriever = HybridRetriever(FixedRetriever(), FixedRetriever())

    assert r is not None
