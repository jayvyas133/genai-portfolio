"""Build the configured ``LLMProvider`` from settings.

This is the one place that maps ``AICORE_PROVIDER`` to an adapter class,
so projects never import a concrete provider directly.
"""

from typing import assert_never

from aicore.config import LLMSettings
from aicore.providers.base import LLMProvider
from aicore.providers.fakeai import FakeProvider
from aicore.providers.ollama import OllamaProvider


def build_provider(settings: LLMSettings) -> LLMProvider:
    """Return the adapter named by ``settings.provider``.

    Args:
        settings: Validated settings; ``provider`` picks the adapter.

    Returns:
        A ready-to-use provider. ``OllamaProvider`` holds an HTTP
        connection pool, so call its ``aclose`` when finished.
    """
    name = settings.provider
    match name:
        case "fake":
            return FakeProvider(model=settings.model)
        case "ollama":
            return OllamaProvider(str(settings.ollama_base_url))
        case _:
            assert_never(name)
