"""Integration tests for ``RedisCache`` against a real Redis.

Start it with
``docker compose -f infra/docker-compose.yml up -d redis``.
Tests skip when Redis is unreachable. They use unique
keys and a short TTL, so no data is ever flushed.
"""

import os
import uuid
from collections.abc import AsyncIterator

import pytest
from aicore.cache import cache_key
from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.models import LLMRequest, LLMResponse, Message, Usage
from aicore.providers.fakeai import FakeProvider
from aicore.redis_cache import RedisCache
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

pytestmark = pytest.mark.integration

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
TTL_S = 60


@pytest.fixture
async def redis_client() -> AsyncIterator[Redis]:
    """Yield a connected client, or skip if Redis is down."""
    client = Redis.from_url(REDIS_URL)
    try:
        await client.ping()
    except RedisConnectionError:
        await client.aclose()
        pytest.skip(f"Redis not reachable at {REDIS_URL}")
    yield client
    await client.aclose()


def unique_request() -> LLMRequest:
    """Build a request no other test run will ever send."""
    msg = Message(role="user", content=uuid.uuid4().hex)
    return LLMRequest(messages=[msg], model="m")


async def test_round_trip_sets_ttl(redis_client: Redis) -> None:
    """A stored reply comes back equal, with an expiry."""
    cache = RedisCache(redis_client, ttl_s=TTL_S)
    key = f"aicore:test:{uuid.uuid4().hex}"
    resp = LLMResponse(
        text="hi",
        model="m",
        usage=Usage(input_tokens=1, output_tokens=1),
    )

    await cache.set(key, resp)

    assert await cache.get(key) == resp
    assert 0 < await redis_client.ttl(key) <= TTL_S


async def test_unknown_key_is_miss(redis_client: Redis) -> None:
    """A key never written returns None."""
    cache = RedisCache(redis_client, ttl_s=TTL_S)
    assert await cache.get(f"aicore:test:{uuid.uuid4().hex}") is None


async def test_clients_share_replies(redis_client: Redis) -> None:
    """A second client, like another process, gets a hit."""
    request = unique_request()
    settings = LLMSettings(model="m")
    first = FakeProvider(script=["shared"])
    second = FakeProvider()

    cache_a = RedisCache(redis_client, ttl_s=TTL_S)
    cache_b = RedisCache(redis_client, ttl_s=TTL_S)
    await LLMClient(first, settings, cache=cache_a).complete(request)
    reply = await LLMClient(second, settings, cache=cache_b).complete(request)

    assert reply.text == "shared"
    assert second.calls == []


async def test_corrupt_entry_is_a_miss(redis_client: Redis) -> None:
    """Bad JSON is logged, the provider answers, and the entry heals."""
    request = unique_request()
    key = cache_key("fake", request)
    await redis_client.set(key, b"not json", ex=TTL_S)
    provider = FakeProvider(script=["fresh"])
    cache = RedisCache(redis_client, ttl_s=TTL_S)
    client = LLMClient(provider, LLMSettings(model="m"), cache=cache)

    reply = await client.complete(request)

    assert reply.text == "fresh"
    assert len(provider.calls) == 1
    healed = await cache.get(key)
    assert healed is not None
    assert healed.text == "fresh"
