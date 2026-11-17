"""Unit tests for ``LLMClient.embed`` and ``LLMClient.rerank``.

``FakeProvider`` stands in for every backend, so no network is needed.
"""

import asyncio

import pytest
from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.errors import (
    AuthenticationError,
    InvalidRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from aicore.models import EmbedRequest, EmbedResponse, LLMRequest, Message, RerankRequest
from aicore.providers.fakeai import FakeProvider
from aicore.tracing import InMemoryTracer

SETTINGS = LLMSettings(
    model="chat-model",
    embed_model="embed-model",
    rerank_model="rerank-model",
    max_retries=2,
)


def make_client(
    fake: FakeProvider,
    tracer: InMemoryTracer | None = None,
    settings: LLMSettings = SETTINGS,
) -> LLMClient:
    """A client using ``fake`` for every capability, with instant retries."""
    return LLMClient(fake, settings, base_delay_s=0, tracer=tracer, embedder=fake, reranker=fake)


class SlowEmbedder:
    """Embedding provider that never answers in time."""

    name = "slow"

    def __init__(self) -> None:
        """Start with no calls."""
        self.calls = 0

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        """Sleep far longer than any test timeout."""
        self.calls += 1
        await asyncio.sleep(10)
        raise AssertionError("unreachable")  # pragma: no cover


class TestEmbed:
    """``embed`` adds model defaults, retries and tracing."""

    async def test_model_defaults_to_settings_embed_model(self) -> None:
        """A request without a model gets ``settings.embed_model``."""
        fake = FakeProvider()

        resp = await make_client(fake).embed(EmbedRequest(texts=["a"]))

        assert fake.embed_calls[0].model == "embed-model"
        assert resp.model == "embed-model"

    async def test_explicit_model_wins(self) -> None:
        """An explicit model on the request is kept."""
        fake = FakeProvider()

        await make_client(fake).embed(EmbedRequest(texts=["a"], model="mine"))

        assert fake.embed_calls[0].model == "mine"

    async def test_retries_transient_errors_then_succeeds(self) -> None:
        """Retryable errors are retried; the trace counts every attempt."""
        fake = FakeProvider(embed_errors=[ProviderUnavailableError("503")] * 2)
        tracer = InMemoryTracer()

        resp = await make_client(fake, tracer).embed(EmbedRequest(texts=["a", "b"]))

        assert len(resp.vectors) == 2
        assert len(fake.embed_calls) == 3
        (rec,) = tracer.records
        assert (rec.operation, rec.attempts, rec.outcome) == ("embed", 3, "ok")
        assert rec.provider == "fake"
        assert rec.model == "embed-model"
        assert rec.cost_usd == 0.0  # fake is a local provider

    async def test_gives_up_after_max_retries(self) -> None:
        """After ``1 + max_retries`` attempts the last error surfaces and is traced."""
        fake = FakeProvider(embed_errors=[ProviderUnavailableError("503")] * 5)
        tracer = InMemoryTracer()

        with pytest.raises(ProviderUnavailableError):
            await make_client(fake, tracer).embed(EmbedRequest(texts=["a"]))

        assert len(fake.embed_calls) == 3
        (rec,) = tracer.records
        assert (rec.attempts, rec.outcome) == (3, "ProviderUnavailableError")
        assert rec.usage is None
        assert rec.cost_usd is None

    async def test_permanent_error_is_not_retried(self) -> None:
        """A non-retryable error fails on the first attempt."""
        fake = FakeProvider(embed_errors=[AuthenticationError("401")])

        with pytest.raises(AuthenticationError):
            await make_client(fake).embed(EmbedRequest(texts=["a"]))

        assert len(fake.embed_calls) == 1

    async def test_each_attempt_has_its_own_timeout(self) -> None:
        """A hung embedder times out on every attempt, then raises."""
        slow = SlowEmbedder()
        settings = SETTINGS.model_copy(update={"timeout_s": 0.01, "max_retries": 1})
        client = LLMClient(FakeProvider(), settings, base_delay_s=0, embedder=slow)

        with pytest.raises(ProviderTimeoutError):
            await client.embed(EmbedRequest(texts=["a"]))

        assert slow.calls == 2

    async def test_without_embedder_raises(self) -> None:
        """A client built without an embedder refuses ``embed`` clearly."""
        client = LLMClient(FakeProvider(), SETTINGS)

        with pytest.raises(InvalidRequestError, match="embedder"):
            await client.embed(EmbedRequest(texts=["a"]))


class TestRerank:
    """``rerank`` adds model defaults, retries and tracing."""

    async def test_model_defaults_and_results_pass_through(self) -> None:
        """Settings supply the model; the provider's ranking is returned."""
        fake = FakeProvider()

        resp = await make_client(fake).rerank(
            RerankRequest(query="gpu memory", documents=["cricket", "gpu memory"])
        )

        assert fake.rerank_calls[0].model == "rerank-model"
        assert [r.index for r in resp.results] == [1, 0]

    async def test_retries_and_traces_operation(self) -> None:
        """A transient error is retried; the record says ``rerank``."""
        fake = FakeProvider(rerank_errors=[ProviderUnavailableError("503")])
        tracer = InMemoryTracer()

        await make_client(fake, tracer).rerank(RerankRequest(query="q", documents=["q"]))

        (rec,) = tracer.records
        assert (rec.operation, rec.attempts, rec.outcome) == ("rerank", 2, "ok")
        assert rec.model == "rerank-model"

    async def test_without_reranker_raises(self) -> None:
        """A client built without a reranker refuses ``rerank`` clearly."""
        client = LLMClient(FakeProvider(), SETTINGS, embedder=FakeProvider())

        with pytest.raises(InvalidRequestError, match="reranker"):
            await client.rerank(RerankRequest(query="q", documents=["d"]))


async def test_complete_records_complete_operation() -> None:
    """Chat calls are traced with ``operation="complete"`` after the refactor."""
    tracer = InMemoryTracer()

    await make_client(FakeProvider(), tracer).complete(
        LLMRequest(messages=[Message(role="user", content="hi")])
    )

    assert tracer.records[0].operation == "complete"
