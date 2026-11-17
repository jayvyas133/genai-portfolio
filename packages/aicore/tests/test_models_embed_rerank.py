"""Validation tests for the embedding and rerank request/response models."""

import pytest
from aicore.models import (
    EmbedRequest,
    EmbedResponse,
    RerankRequest,
    RerankResponse,
    RerankResult,
    Usage,
)
from pydantic import ValidationError

ZERO_USAGE = Usage(input_tokens=0, output_tokens=0)


class TestEmbedRequest:
    """``EmbedRequest`` accepts a non-empty batch of non-empty texts."""

    def test_defaults_to_document_input_type(self) -> None:
        """Ingestion is the common case, so ``input_type`` defaults to document."""
        req = EmbedRequest(texts=["a"])
        assert req.input_type == "document"
        assert req.model is None

    def test_accepts_query_input_type(self) -> None:
        """Queries are flagged explicitly so adapters can add a query prefix."""
        assert EmbedRequest(texts=["a"], input_type="query").input_type == "query"

    def test_rejects_unknown_input_type(self) -> None:
        """Only the two retrieval sides are allowed."""
        with pytest.raises(ValidationError):
            EmbedRequest(texts=["a"], input_type="passage")  # type: ignore[arg-type]

    def test_rejects_empty_batch(self) -> None:
        """An empty batch would be a wasted provider call."""
        with pytest.raises(ValidationError):
            EmbedRequest(texts=[])

    def test_rejects_empty_string_in_batch(self) -> None:
        """Empty strings produce meaningless vectors, so they are refused early."""
        with pytest.raises(ValidationError):
            EmbedRequest(texts=["ok", ""])


class TestEmbedResponse:
    """``EmbedResponse.dimensions`` reports the vector length."""

    def test_dimensions_from_first_vector(self) -> None:
        """Dimensions equal the length of each vector."""
        resp = EmbedResponse(vectors=[[0.1, 0.2, 0.3]], model="m", usage=ZERO_USAGE)
        assert resp.dimensions == 3

    def test_dimensions_zero_when_empty(self) -> None:
        """No vectors means zero dimensions rather than an IndexError."""
        resp = EmbedResponse(vectors=[], model="m", usage=ZERO_USAGE)
        assert resp.dimensions == 0


class TestRerankRequest:
    """``RerankRequest`` needs a query, documents and a positive ``top_n``."""

    def test_valid_request_defaults(self) -> None:
        """``top_n`` and ``model`` default to ``None``."""
        req = RerankRequest(query="q", documents=["d1", "d2"])
        assert req.top_n is None
        assert req.model is None

    def test_rejects_empty_query(self) -> None:
        """A blank query cannot be scored against anything."""
        with pytest.raises(ValidationError):
            RerankRequest(query="", documents=["d"])

    def test_rejects_empty_documents(self) -> None:
        """At least one candidate document is required."""
        with pytest.raises(ValidationError):
            RerankRequest(query="q", documents=[])

    def test_rejects_empty_document_string(self) -> None:
        """Every candidate document must have content."""
        with pytest.raises(ValidationError):
            RerankRequest(query="q", documents=["d", ""])

    @pytest.mark.parametrize("top_n", [0, -1])
    def test_rejects_non_positive_top_n(self, top_n: int) -> None:
        """``top_n`` must be at least 1."""
        with pytest.raises(ValidationError):
            RerankRequest(query="q", documents=["d"], top_n=top_n)


class TestRerankResult:
    """``RerankResult`` points back into the request's documents."""

    def test_rejects_negative_index(self) -> None:
        """Indexes are positions in a list, so they cannot be negative."""
        with pytest.raises(ValidationError):
            RerankResult(index=-1, score=0.5)

    def test_response_holds_results(self) -> None:
        """A response carries results, model and usage unchanged."""
        resp = RerankResponse(
            results=[RerankResult(index=1, score=0.9), RerankResult(index=0, score=0.1)],
            model="m",
            usage=ZERO_USAGE,
        )
        assert [r.index for r in resp.results] == [1, 0]
