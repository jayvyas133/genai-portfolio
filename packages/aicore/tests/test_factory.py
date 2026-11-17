"""Tests for ``build_provider`` and ``build_client``: config picks the adapter."""

import pytest
from aicore.config import LLMSettings, ProviderName
from aicore.errors import InvalidRequestError
from aicore.models import EmbedRequest, RerankRequest
from aicore.providers.cloudflare import CloudflareProvider
from aicore.providers.factory import build_client, build_provider
from aicore.providers.fakeai import FakeProvider
from aicore.providers.ollama import OllamaProvider
from aicore.tracing import InMemoryTracer
from pydantic import SecretStr


@pytest.mark.parametrize(
    ("name", "expected"),
    [("fake", FakeProvider), ("ollama", OllamaProvider)],
)
def test_build_provider_picks_adapter(name: ProviderName, expected: type[object]) -> None:
    """``settings.provider`` decides which adapter is built."""
    provider = build_provider(LLMSettings(provider=name))

    assert isinstance(provider, expected)


def test_build_provider_picks_cloudflare() -> None:
    """With token and account ID set, Cloudflare is built."""
    settings = LLMSettings(
        provider="cloudflare",
        api_key=SecretStr("tok"),
        cloudflare_account_id="acct",
    )

    provider = build_provider(settings)

    assert isinstance(provider, CloudflareProvider)


@pytest.mark.parametrize(
    ("api_key", "account_id", "missing"),
    [
        (None, "acct", "AICORE_API_KEY"),
        (SecretStr(""), "acct", "AICORE_API_KEY"),
        (SecretStr("tok"), None, "AICORE_CLOUDFLARE_ACCOUNT_ID"),
    ],
)
def test_cloudflare_fails_fast_without_credentials(
    api_key: SecretStr | None,
    account_id: str | None,
    missing: str,
) -> None:
    """Missing or empty creds raise at build time,
    naming the env var to set."""
    settings = LLMSettings(
        provider="cloudflare",
        api_key=api_key,
        cloudflare_account_id=account_id,
    )

    with pytest.raises(ValueError, match=missing):
        build_provider(settings)


async def test_build_client_fake_serves_embed_and_rerank() -> None:
    """The fake adapter backs chat, embed and rerank, and calls are traced."""
    tracer = InMemoryTracer()
    client = build_client(LLMSettings(_env_file=None, provider="fake"), tracer=tracer)

    embedded = await client.embed(EmbedRequest(texts=["a"]))
    reranked = await client.rerank(RerankRequest(query="q", documents=["d"]))
    await client.aclose()

    assert len(embedded.vectors) == 1
    assert len(reranked.results) == 1
    assert [r.operation for r in tracer.records] == ["embed", "rerank"]


async def test_build_client_ollama_has_no_embedder() -> None:
    """Ollama only chats, so ``embed`` fails clearly instead of silently."""
    client = build_client(LLMSettings(_env_file=None, provider="ollama"))

    with pytest.raises(InvalidRequestError, match="without an embedder"):
        await client.embed(EmbedRequest(texts=["a"]))
    await client.aclose()


async def test_build_client_cloudflare_builds_and_closes() -> None:
    """With credentials set, a Cloudflare client is built without any network call."""
    settings = LLMSettings(
        _env_file=None,
        provider="cloudflare",
        api_key=SecretStr("tok"),
        cloudflare_account_id="acct",
    )

    await build_client(settings).aclose()


def test_build_client_cloudflare_fails_fast_without_credentials() -> None:
    """The same credential check as ``build_provider`` applies."""
    settings = LLMSettings(_env_file=None, provider="cloudflare", cloudflare_account_id="acct")

    with pytest.raises(ValueError, match="AICORE_API_KEY"):
        build_client(settings)
