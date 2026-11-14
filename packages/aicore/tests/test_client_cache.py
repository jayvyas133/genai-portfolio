"""Unit tests for response caching in ``LLMClient``."""

import pytest
from aicore.cache import InMemoryCache
from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.errors import AuthenticationError
from aicore.models import LLMRequest, LLMResponse, Message
from aicore.providers.fakeai import FakeProvider, Outcome
from aicore.tracing import InMemoryTracer


def make_request(temperature: float = 0.0) -> LLMRequest:
    """Build a one-message request."""
    return LLMRequest(
        messages=[Message(role="user", content="hello")],
        temperature=temperature,
    )


def make_client(
    script: list[Outcome] | None = None,
    cache: InMemoryCache | None = None,
    model: str = "m",
) -> tuple[LLMClient, FakeProvider, InMemoryTracer]:
    """Return a cached client, its provider and its tracer."""
    provider = FakeProvider(script=script or [])
    tracer = InMemoryTracer()
    client = LLMClient(
        provider,
        LLMSettings(model=model),
        base_delay_s=0,
        tracer=tracer,
        cache=cache if cache is not None else InMemoryCache(),
    )
    return client, provider, tracer


class BrokenCache:
    """Cache whose every operation fails, like Redis down."""

    async def get(self, key: str) -> LLMResponse | None:
        """Raise, to simulate a failed read."""
        raise ConnectionError("cache down")

    async def set(self, key: str, response: LLMResponse) -> None:
        """Raise, to simulate a failed write."""
        raise ConnectionError("cache down")


async def test_hit_skips_provider() -> None:
    """The second identical call is served from the cache."""
    client, provider, _ = make_client(["first"])

    a = await client.complete(make_request())
    b = await client.complete(make_request())

    assert a == b
    assert len(provider.calls) == 1


async def test_hit_is_traced_as_cached() -> None:
    """A hit: cached, no attempts, free, usage kept."""
    client, _, tracer = make_client(["first reply"])

    await client.complete(make_request())
    await client.complete(make_request())

    miss, hit = tracer.records
    assert miss.cached is False
    assert miss.attempts == 1
    assert hit.cached is True
    assert hit.attempts == 0
    assert hit.cost_usd == 0.0
    assert hit.usage == miss.usage


async def test_sampled_requests_bypass_cache() -> None:
    """``temperature > 0`` always calls the provider."""
    client, provider, tracer = make_client()

    await client.complete(make_request(temperature=0.7))
    await client.complete(make_request(temperature=0.7))

    assert len(provider.calls) == 2
    assert not any(r.cached for r in tracer.records)


async def test_errors_are_not_cached() -> None:
    """A failed call stores nothing; the next one retries."""
    client, provider, _ = make_client([AuthenticationError("bad key")])

    with pytest.raises(AuthenticationError):
        await client.complete(make_request())
    await client.complete(make_request())  # miss: real call
    await client.complete(make_request())  # hit

    assert len(provider.calls) == 2


async def test_model_change_is_a_miss() -> None:
    """Changing the default model never reuses old replies."""
    shared = InMemoryCache()
    old, old_provider, _ = make_client(cache=shared, model="m1")
    new, new_provider, _ = make_client(cache=shared, model="m2")

    await old.complete(make_request())
    await new.complete(make_request())

    assert len(old_provider.calls) == 1
    assert len(new_provider.calls) == 1


async def test_broken_cache_does_not_break_call() -> None:
    """Cache errors are logged; the provider still answers."""
    provider = FakeProvider(script=["still works"])
    client = LLMClient(
        provider,
        LLMSettings(model="m"),
        cache=BrokenCache(),
    )

    reply = await client.complete(make_request())

    assert reply.text == "still works"
    assert len(provider.calls) == 1
