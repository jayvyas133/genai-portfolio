"""Measure latency and cost of the configured provider.

Runs the same short prompt N times through
``LLMClient`` and prints one row per call plus a
summary, using the records from M2 tracing.

If ``REDIS_URL`` is set, a ``RedisCache`` is used:
call 1 is a miss (the prompt carries a per-run tag),
later calls are hits, and the summary compares them.

Usage:
    uv run python scripts/measure_llm.py 5
"""

import asyncio
import os
import statistics
import sys
import uuid

from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.models import LLMRequest, Message
from aicore.providers.factory import build_provider
from aicore.redis_cache import RedisCache
from aicore.tracing import CallRecord, InMemoryTracer

PROMPT = "In one sentence, what is a vector database?"


def median_ms(records: list[CallRecord]) -> str:
    """Median latency of ``records`` as text, or '-' if empty."""
    if not records:
        return "-"
    return f"{statistics.median(r.latency_ms for r in records):.1f} ms"


async def main(n: int) -> None:
    """Run ``n`` calls and print their records."""
    settings = LLMSettings()
    provider = build_provider(settings)
    tracer = InMemoryTracer()
    redis_url = os.environ.get("REDIS_URL")
    cache = RedisCache.from_url(redis_url, settings.cache_ttl_s) if redis_url else None
    client = LLMClient(provider, settings, tracer=tracer, cache=cache)

    # A fresh tag per run, so call 1 is a real miss even
    # when an earlier run left this prompt in Redis.
    tag = uuid.uuid4().hex[:8]
    request = LLMRequest(
        messages=[Message(role="user", content=f"{PROMPT} [run {tag}]")],
        max_tokens=64,
    )

    try:
        for _ in range(n):
            await client.complete(request)
    finally:
        if cache is not None:
            await cache.aclose()

    print(
        f"provider={settings.provider} model={settings.model} cache={'redis' if cache else 'off'}"
    )
    for i, rec in enumerate(tracer.records, start=1):
        u = rec.usage
        print(
            f"{i}: {rec.latency_ms:8.1f} ms "
            f"cached={rec.cached!s:5} "
            f"in={u.input_tokens if u else '-'} "
            f"out={u.output_tokens if u else '-'} "
            f"neurons={u.neurons if u else '-'} "
            f"cost_usd={rec.cost_usd}"
        )

    misses = [r for r in tracer.records if not r.cached]
    hits = [r for r in tracer.records if r.cached]
    print(f"misses={len(misses)} median={median_ms(misses)}")
    print(f"hits={len(hits)} median={median_ms(hits)}")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 5))
