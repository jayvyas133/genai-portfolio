"""Tests for ``build_provider``: config picks the adapter."""

import pytest
from aicore.config import LLMSettings, ProviderName
from aicore.providers.cloudflare import CloudflareProvider
from aicore.providers.factory import build_provider
from aicore.providers.fakeai import FakeProvider
from aicore.providers.ollama import OllamaProvider
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
