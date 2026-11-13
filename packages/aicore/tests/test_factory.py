"""Tests for ``build_provider``: config picks the adapter."""

import pytest
from aicore.config import LLMSettings, ProviderName
from aicore.providers.factory import build_provider
from aicore.providers.fakeai import FakeProvider
from aicore.providers.ollama import OllamaProvider


@pytest.mark.parametrize(
    ("name", "expected"),
    [("fake", FakeProvider), ("ollama", OllamaProvider)],
)
def test_build_provider_picks_adapter(name: ProviderName, expected: type[object]) -> None:
    """``settings.provider`` decides which adapter is built."""
    provider = build_provider(LLMSettings(provider=name))

    assert isinstance(provider, expected)
