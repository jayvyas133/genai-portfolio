"""Redis backend for the response cache.

Values are ``LLMResponse`` JSON; Redis expires them
after ``ttl_s`` seconds. Shared by every process that
points at the same Redis, and survives restarts.
"""

from typing import Self

from redis.asyncio import Redis

from aicore.models import LLMResponse


class RedisCache:
    """``ResponseCache`` backed by Redis.

    Errors (connection refused, bad JSON) are raised;
    ``LLMClient`` logs them and treats them as a miss.
    """

    def __init__(self, client: Redis, ttl_s: int) -> None:
        """Wrap an existing async Redis client.

        Args:
            client: Connected ``redis.asyncio.Redis``.
            ttl_s: Seconds each entry lives; must be > 0.
        """
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        self._redis = client
        self._ttl_s = ttl_s

    @classmethod
    def from_url(cls, url: str, ttl_s: int) -> Self:
        """Create a cache from a URL like ``redis://host:6379/0``."""
        return cls(Redis.from_url(url), ttl_s)

    async def get(self, key: str) -> LLMResponse | None:
        """Return the stored response, or None on a miss."""
        raw = await self._redis.get(key)
        if raw is None:
            return None
        return LLMResponse.model_validate_json(raw)

    async def set(self, key: str, response: LLMResponse) -> None:
        """Store ``response`` with the TTL, in one command."""
        await self._redis.set(key, response.model_dump_json(), ex=self._ttl_s)

    async def aclose(self) -> None:
        """Close the connection pool; call on app shutdown."""
        await self._redis.aclose()
