"""Tests for ``LLMClient.aclose``: every backend closed exactly once."""

from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.providers.fakeai import FakeProvider


class ClosingFake(FakeProvider):
    """``FakeProvider`` that counts ``aclose`` calls, like an HTTP adapter."""

    def __init__(self) -> None:
        """Start with zero closes."""
        super().__init__()
        self.closed = 0

    async def aclose(self) -> None:
        """Record the close."""
        self.closed += 1


SETTINGS = LLMSettings(_env_file=None)  # ignore any local .env


async def test_shared_backend_closed_once() -> None:
    """One adapter serving chat, embed and rerank is closed a single time."""
    shared = ClosingFake()
    client = LLMClient(shared, SETTINGS, embedder=shared, reranker=shared)

    await client.aclose()

    assert shared.closed == 1


async def test_distinct_backends_each_closed() -> None:
    """Separate chat and embedding backends are both closed."""
    chat, embed = ClosingFake(), ClosingFake()
    client = LLMClient(chat, SETTINGS, embedder=embed)

    await client.aclose()

    assert (chat.closed, embed.closed) == (1, 1)


async def test_backends_without_aclose_are_skipped() -> None:
    """``FakeProvider`` has no pool; closing is a harmless no-op."""
    await LLMClient(FakeProvider(), SETTINGS).aclose()
