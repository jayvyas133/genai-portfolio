"""Tests for ``CloudflareProvider.embed`` and ``CloudflareProvider.rerank``.

An ``httpx.MockTransport`` stands in for the Workers AI API, so these tests
need no network or account.
"""

import json
from collections.abc import Callable

import httpx
import pytest
from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.errors import (
    InvalidRequestError,
    ProviderTimeoutError,
    QuotaExhaustedError,
    RateLimitError,
)
from aicore.models import EmbedRequest, RerankRequest
from aicore.providers.base import EmbeddingProvider, RerankProvider
from aicore.providers.cloudflare import CloudflareProvider

Handler = Callable[[httpx.Request], httpx.Response]

ACCOUNT = "acct-123"
TOKEN = "test-token"
EMBED_MODEL = "@cf/baai/bge-base-en-v1.5"
RERANK_MODEL = "@cf/baai/bge-reranker-base"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def ok(result: object) -> httpx.Response:
    """Wrap ``result`` in Cloudflare's success envelope."""
    return httpx.Response(
        200, json={"result": result, "success": True, "errors": [], "messages": []}
    )


def embed_ok(vectors: list[list[float]]) -> httpx.Response:
    """A successful embedding response holding ``vectors``."""
    return ok({"shape": [len(vectors), len(vectors[0])], "data": vectors})


def recording(*responses: httpx.Response) -> tuple[CloudflareProvider, list[httpx.Request]]:
    """Return a provider replaying ``responses`` and the requests it saw."""
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the request; answer with the next queued response."""
        seen.append(request)
        return queue.pop(0)

    provider = CloudflareProvider(ACCOUNT, TOKEN, transport=httpx.MockTransport(handler))
    return provider, seen


def test_cloudflare_satisfies_new_protocols() -> None:
    """mypy checks these assignments; the test makes the intent explicit."""
    provider = CloudflareProvider(ACCOUNT, TOKEN)
    embedder: EmbeddingProvider = provider
    reranker: RerankProvider = provider
    assert embedder is reranker


class TestEmbed:
    """Embedding payloads, the BGE query prefix and response mapping."""

    async def test_documents_sent_unprefixed_and_vectors_mapped(self) -> None:
        """Documents go out as-is; vectors come back in order."""
        provider, seen = recording(embed_ok([[0.1, 0.2], [0.3, 0.4]]))

        resp = await provider.embed(EmbedRequest(texts=["a", "b"], model=EMBED_MODEL))

        assert seen[0].url.path == f"/client/v4/accounts/{ACCOUNT}/ai/run/{EMBED_MODEL}"
        assert json.loads(seen[0].content) == {"text": ["a", "b"]}
        assert resp.vectors == [[0.1, 0.2], [0.3, 0.4]]
        assert resp.dimensions == 2
        assert resp.model == EMBED_MODEL
        assert resp.usage.total_tokens == 0

    async def test_reports_input_tokens_from_usage(self) -> None:
        """``result.usage.prompt_tokens`` becomes ``input_tokens``."""
        result = {
            "shape": [1, 1],
            "data": [[0.5]],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
        }
        provider, _ = recording(ok(result))

        resp = await provider.embed(EmbedRequest(texts=["a"], model=EMBED_MODEL))

        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 0
        assert resp.usage.neurons is None

    async def test_queries_get_bge_prefix(self) -> None:
        """BGE queries carry the retrieval instruction."""
        provider, seen = recording(embed_ok([[0.1]]))

        await provider.embed(
            EmbedRequest(texts=["what is CUDA?"], input_type="query", model=EMBED_MODEL)
        )

        assert json.loads(seen[0].content) == {"text": [QUERY_PREFIX + "what is CUDA?"]}

    async def test_non_bge_query_is_not_prefixed(self) -> None:
        """The instruction is BGE-specific; other models get raw text."""
        provider, seen = recording(embed_ok([[0.1]]))

        await provider.embed(
            EmbedRequest(texts=["q"], input_type="query", model="@cf/other/embedder")
        )

        assert json.loads(seen[0].content) == {"text": ["q"]}

    async def test_missing_model_raises_without_calling_api(self) -> None:
        """No model means no URL, so the request is rejected up front."""
        provider, seen = recording()

        with pytest.raises(InvalidRequestError):
            await provider.embed(EmbedRequest(texts=["a"]))
        assert seen == []

    async def test_quota_error_is_mapped(self) -> None:
        """Error mapping is shared with ``complete`` via ``_run``."""
        body = {
            "result": None,
            "success": False,
            "errors": [{"code": 3036, "message": "daily limit"}],
        }
        provider, _ = recording(httpx.Response(429, json=body))

        with pytest.raises(QuotaExhaustedError):
            await provider.embed(EmbedRequest(texts=["a"], model=EMBED_MODEL))

    async def test_timeout_is_mapped(self) -> None:
        """Transport timeouts become ``ProviderTimeoutError``."""

        def handler(request: httpx.Request) -> httpx.Response:
            """Simulate a read timeout."""
            raise httpx.ReadTimeout("slow", request=request)

        provider = CloudflareProvider(ACCOUNT, TOKEN, transport=httpx.MockTransport(handler))

        with pytest.raises(ProviderTimeoutError):
            await provider.embed(EmbedRequest(texts=["a"], model=EMBED_MODEL))


class TestRerank:
    """Rerank payloads, re-sorting, truncation and error mapping."""

    async def test_payload_without_top_n(self) -> None:
        """Documents become ``contexts``; ``top_k`` is omitted when unset."""
        provider, seen = recording(ok({"response": [{"id": 0, "score": 0.5}]}))

        await provider.rerank(RerankRequest(query="q", documents=["d0"], model=RERANK_MODEL))

        assert seen[0].url.path.endswith(RERANK_MODEL)
        assert json.loads(seen[0].content) == {"query": "q", "contexts": [{"text": "d0"}]}

    async def test_payload_with_top_n(self) -> None:
        """``top_n`` is sent as Cloudflare's ``top_k``."""
        provider, seen = recording(ok({"response": [{"id": 0, "score": 0.5}]}))

        await provider.rerank(
            RerankRequest(query="q", documents=["d0", "d1"], top_n=1, model=RERANK_MODEL)
        )

        assert json.loads(seen[0].content)["top_k"] == 1

    async def test_results_resorted_by_score(self) -> None:
        """Unordered API output is sorted highest score first, ids become indexes."""
        unordered = [
            {"id": 0, "score": 0.1},
            {"id": 2, "score": 0.9},
            {"id": 1, "score": 0.5},
        ]
        provider, _ = recording(ok({"response": unordered}))

        resp = await provider.rerank(
            RerankRequest(query="q", documents=["a", "b", "c"], model=RERANK_MODEL)
        )

        assert [(r.index, r.score) for r in resp.results] == [(2, 0.9), (1, 0.5), (0, 0.1)]
        assert resp.model == RERANK_MODEL
        assert resp.usage.total_tokens == 0

    async def test_top_n_enforced_even_if_api_returns_more(self) -> None:
        """Truncation happens client-side too, so the contract always holds."""
        items = [{"id": i, "score": s} for i, s in enumerate([0.2, 0.8, 0.5])]
        provider, _ = recording(ok({"response": items}))

        resp = await provider.rerank(
            RerankRequest(query="q", documents=["a", "b", "c"], top_n=2, model=RERANK_MODEL)
        )

        assert [r.index for r in resp.results] == [1, 2]

    async def test_missing_model_raises_without_calling_api(self) -> None:
        """No model means no URL, so the request is rejected up front."""
        provider, seen = recording()

        with pytest.raises(InvalidRequestError):
            await provider.rerank(RerankRequest(query="q", documents=["d"]))
        assert seen == []

    async def test_rate_limit_is_mapped(self) -> None:
        """HTTP 429 without code 3036 is a retryable ``RateLimitError``."""
        provider, _ = recording(httpx.Response(429, headers={"Retry-After": "2"}))

        with pytest.raises(RateLimitError):
            await provider.rerank(RerankRequest(query="q", documents=["d"], model=RERANK_MODEL))

    async def test_success_false_raises(self) -> None:
        """A 200 with ``success: false`` is still a failure."""
        body = {"result": None, "success": False, "errors": [{"code": 5006, "message": "bad"}]}
        provider, _ = recording(httpx.Response(200, json=body))

        with pytest.raises(InvalidRequestError, match="5006"):
            await provider.rerank(RerankRequest(query="q", documents=["d"], model=RERANK_MODEL))


@pytest.mark.integration
async def test_real_cloudflare_embed_and_rerank() -> None:
    """Call the real Workers AI embedding and rerank models (a few neurons).

    Confirms the request and response field names against the live API, and
    that relevance points the right way. Skipped unless ``AICORE_API_KEY``
    and ``AICORE_CLOUDFLARE_ACCOUNT_ID`` are set.
    """
    settings = LLMSettings(embed_model=EMBED_MODEL, rerank_model=RERANK_MODEL, max_retries=0)
    if settings.api_key is None or not settings.cloudflare_account_id:
        pytest.skip("Cloudflare credentials not set")

    provider = CloudflareProvider(
        settings.cloudflare_account_id, settings.api_key.get_secret_value()
    )
    client = LLMClient(provider, settings, embedder=provider, reranker=provider)
    query = "How much memory does a GPU have?"
    docs = [
        "The H100 GPU ships with 80 GB of high-bandwidth memory.",
        "Cricket is played between two teams of eleven players.",
    ]
    try:
        q = await client.embed(EmbedRequest(texts=[query], input_type="query"))
        d = await client.embed(EmbedRequest(texts=docs))
        ranked = await client.rerank(RerankRequest(query=query, documents=docs))
    finally:
        await provider.aclose()

    assert q.dimensions == d.dimensions == 768
    assert q.usage.input_tokens > 0
    sims = [sum(a * b for a, b in zip(q.vectors[0], v, strict=True)) for v in d.vectors]
    assert sims[0] > sims[1]
    assert [r.index for r in ranked.results] == [0, 1]
