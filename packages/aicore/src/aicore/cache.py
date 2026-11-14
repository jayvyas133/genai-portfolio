"""Response caching for ``LLMClient``.

A cache maps a request key (see ``cache_key``) to a
previous ``LLMResponse``. Backends implement the small
``ResponseCache`` protocol, so the client works the same
with an in-memory dict in tests and Redis in production.
"""

import hashlib
import json
from typing import Protocol

from aicore.models import LLMRequest, LLMResponse

KEY_PREFIX = "aicore:v1:"
"""Bump the version to invalidate every old cache entry."""


def cache_key(provider: str, request: LLMRequest) -> str:
    """Return a stable cache key for ``request`` on ``provider``.

    The whole request is serialized with sorted keys, so
    any field that can change the reply changes the key.
    The client must fill in ``request.model`` first;
    otherwise a changed default model would reuse old keys.

    Args:
        provider: Name of the provider that will answer.
        request: The fully resolved request.

    Returns:
        ``KEY_PREFIX`` followed by a SHA-256 hex digest.
    """
    payload = {
        "provider": provider,
        "request": request.model_dump(mode="json"),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return KEY_PREFIX + digest


class ResponseCache(Protocol):
    """Anything that can store and return LLM responses.

    Expiry (TTL, time to live) is the backend's concern
    and is configured when the cache is created.
    """

    async def get(self, key: str) -> LLMResponse | None:
        """Return the cached response for ``key``, or None."""
        ...

    async def set(self, key: str, response: LLMResponse) -> None:
        """Store ``response`` under ``key``."""
        ...


class InMemoryCache:
    """Dict-backed cache for tests and local runs.

    Entries never expire and are lost when the process
    exits. Not shared between processes.
    """

    def __init__(self) -> None:
        """Create an empty cache."""
        self._data: dict[str, LLMResponse] = {}

    async def get(self, key: str) -> LLMResponse | None:
        """Return the stored response, or None on a miss."""
        return self._data.get(key)

    async def set(self, key: str, response: LLMResponse) -> None:
        """Store ``response``, replacing any old entry."""
        self._data[key] = response
