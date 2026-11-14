"""Unit tests for the response caches."""

import pytest
from aicore.cache import KEY_PREFIX, InMemoryCache, cache_key
from aicore.config import LLMSettings
from aicore.models import LLMRequest, LLMResponse, Message, Usage
from pydantic import ValidationError


def make_response(text: str = "hi") -> LLMResponse:
    """Build a small response for cache tests."""
    return LLMResponse(
        text=text,
        model="m",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


async def test_miss_returns_none() -> None:
    """An unknown key is a miss."""
    cache = InMemoryCache()
    assert await cache.get("nope") is None


async def test_set_then_get_returns_response() -> None:
    """A stored response comes back unchanged."""
    cache = InMemoryCache()
    resp = make_response()
    await cache.set("k", resp)
    assert await cache.get("k") == resp


async def test_set_overwrites() -> None:
    """A second set for the same key replaces the first."""
    cache = InMemoryCache()
    await cache.set("k", make_response("old"))
    await cache.set("k", make_response("new"))
    got = await cache.get("k")
    assert got is not None
    assert got.text == "new"


def make_request(text: str = "hello", **kw: object) -> LLMRequest:
    """Build a one-message request; ``kw`` sets other fields."""
    msgs = [Message(role="user", content=text)]
    return LLMRequest.model_validate({"messages": msgs, "model": "m", **kw})


def test_key_is_stable() -> None:
    """Equal requests give equal keys."""
    assert cache_key("fake", make_request()) == cache_key("fake", make_request())


def test_key_has_prefix() -> None:
    """Keys start with the versioned prefix."""
    assert cache_key("fake", make_request()).startswith(KEY_PREFIX)


def test_key_changes_with_content() -> None:
    """A different prompt gives a different key."""
    a = cache_key("fake", make_request("hello"))
    b = cache_key("fake", make_request("bye"))
    assert a != b


def test_key_changes_with_provider() -> None:
    """The same request on another provider gives a different key."""
    req = make_request()
    assert cache_key("fake", req) != cache_key("ollama", req)


def test_key_changes_with_max_tokens() -> None:
    """Generation settings are part of the key."""
    a = cache_key("fake", make_request(max_tokens=10))
    b = cache_key("fake", make_request(max_tokens=20))
    assert a != b


def test_ttl_defaults_to_one_day() -> None:
    """Without config, entries live 24 hours."""
    assert LLMSettings().cache_ttl_s == 86_400


def test_ttl_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``AICORE_CACHE_TTL_S`` sets the TTL."""
    monkeypatch.setenv("AICORE_CACHE_TTL_S", "60")
    assert LLMSettings().cache_ttl_s == 60


def test_ttl_must_be_positive() -> None:
    """A zero TTL is rejected at startup."""
    with pytest.raises(ValidationError):
        LLMSettings(cache_ttl_s=0)
