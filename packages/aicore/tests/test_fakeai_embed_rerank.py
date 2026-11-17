"""Unit tests for ``FakeProvider.embed`` and ``FakeProvider.rerank``."""

import math

import pytest
from aicore.errors import ProviderUnavailableError
from aicore.models import EmbedRequest, RerankRequest
from aicore.providers.base import EmbeddingProvider, LLMProvider, RerankProvider
from aicore.providers.fakeai import FakeProvider


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two already-normalized vectors (their dot product)."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_fake_satisfies_all_protocols() -> None:
    """mypy checks these assignments; the test makes the intent explicit."""
    fake = FakeProvider()
    llm: LLMProvider = fake
    embedder: EmbeddingProvider = fake
    reranker: RerankProvider = fake
    assert llm is embedder is reranker


class TestEmbed:
    """Hashing-trick embeddings are deterministic, normalized and word-sensitive."""

    async def test_vectors_have_configured_dimensions_and_unit_length(self) -> None:
        """Every vector has ``dimensions`` entries and L2 norm 1."""
        resp = await FakeProvider(dimensions=32).embed(EmbedRequest(texts=["GPU memory", "CUDA"]))
        assert resp.dimensions == 32
        assert len(resp.vectors) == 2
        for vector in resp.vectors:
            assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0)

    async def test_deterministic_across_instances(self) -> None:
        """The same text always maps to the same vector (no per-process salt)."""
        req = EmbedRequest(texts=["tensor cores accelerate matrix multiply"])
        first = await FakeProvider().embed(req)
        second = await FakeProvider().embed(req)
        assert first.vectors == second.vectors

    async def test_shared_words_score_higher_than_unrelated(self) -> None:
        """Overlapping texts are closer than texts with no words in common."""
        resp = await FakeProvider().embed(
            EmbedRequest(
                texts=[
                    "nvidia gpu memory bandwidth",
                    "gpu memory bandwidth limits",
                    "cricket match in mumbai",
                ]
            )
        )
        anchor, related, unrelated = resp.vectors
        assert cosine(anchor, related) > cosine(anchor, unrelated)

    async def test_case_and_punctuation_are_ignored(self) -> None:
        """Tokenization lower-cases and drops punctuation."""
        resp = await FakeProvider().embed(EmbedRequest(texts=["GPU, Memory!", "gpu memory"]))
        assert resp.vectors[0] == resp.vectors[1]

    async def test_text_without_words_is_zero_vector(self) -> None:
        """No words means nothing to hash, and no division by zero."""
        resp = await FakeProvider(dimensions=8).embed(EmbedRequest(texts=["!!!"]))
        assert resp.vectors == [[0.0] * 8]

    async def test_usage_and_model(self) -> None:
        """Input tokens are word counts, output tokens are 0, model falls back."""
        resp = await FakeProvider(model="fake-embed").embed(EmbedRequest(texts=["a b", "c"]))
        assert resp.usage.input_tokens == 3
        assert resp.usage.output_tokens == 0
        assert resp.model == "fake-embed"

    async def test_request_model_overrides_default(self) -> None:
        """An explicit ``model`` on the request is reported back."""
        resp = await FakeProvider().embed(EmbedRequest(texts=["a"], model="custom"))
        assert resp.model == "custom"

    async def test_scripted_errors_raise_then_recover(self) -> None:
        """Errors are raised in order, then real results resume; calls are recorded."""
        fake = FakeProvider(embed_errors=[ProviderUnavailableError("down")])
        req = EmbedRequest(texts=["a"])
        with pytest.raises(ProviderUnavailableError):
            await fake.embed(req)
        resp = await fake.embed(req)
        assert len(resp.vectors) == 1
        assert fake.embed_calls == [req, req]


class TestRerank:
    """Word-overlap reranking orders, ties and truncates correctly."""

    DOCS = [
        "cricket scores today",  # 0: no overlap
        "gpu memory bandwidth",  # 1: full overlap
        "gpu prices",  # 2: partial overlap
    ]

    async def test_sorted_by_overlap_score(self) -> None:
        """The best-matching document comes first, with the right scores."""
        resp = await FakeProvider().rerank(
            RerankRequest(query="GPU memory bandwidth", documents=self.DOCS)
        )
        assert [r.index for r in resp.results] == [1, 2, 0]
        assert [r.score for r in resp.results] == pytest.approx([1.0, 1 / 3, 0.0])

    async def test_ties_keep_original_order(self) -> None:
        """Equal scores preserve the input order (stable sort)."""
        resp = await FakeProvider().rerank(
            RerankRequest(query="gpu", documents=["gpu a", "other", "gpu b"])
        )
        assert [r.index for r in resp.results] == [0, 2, 1]

    async def test_top_n_truncates(self) -> None:
        """Only the ``top_n`` best results come back."""
        resp = await FakeProvider().rerank(
            RerankRequest(query="gpu memory bandwidth", documents=self.DOCS, top_n=2)
        )
        assert [r.index for r in resp.results] == [1, 2]

    async def test_query_without_words_scores_zero(self) -> None:
        """A punctuation-only query scores every document 0 without dividing by zero."""
        resp = await FakeProvider().rerank(RerankRequest(query="?!", documents=["a", "b"]))
        assert [r.score for r in resp.results] == [0.0, 0.0]

    async def test_usage_counts_query_and_documents(self) -> None:
        """Input tokens cover the query plus every document; output tokens are 0."""
        resp = await FakeProvider().rerank(RerankRequest(query="a b", documents=["c", "d e"]))
        assert resp.usage.input_tokens == 5
        assert resp.usage.output_tokens == 0

    async def test_scripted_errors_raise_then_recover(self) -> None:
        """Errors are raised in order, then real results resume; calls are recorded."""
        fake = FakeProvider(rerank_errors=[ProviderUnavailableError("down")])
        req = RerankRequest(query="q", documents=["q"])
        with pytest.raises(ProviderUnavailableError):
            await fake.rerank(req)
        resp = await fake.rerank(req)
        assert resp.results[0].score == 1.0
        assert fake.rerank_calls == [req, req]
