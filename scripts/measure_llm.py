"""Measure latency and cost of the configured provider.

Runs the same short prompt N times through
``LLMClient`` and prints one row per call plus a
summary, using the records from M2 tracing.

Usage:
    uv run python scripts/measure_llm.py 5
"""

import asyncio
import statistics
import sys

from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.models import LLMRequest, Message
from aicore.providers.factory import build_provider
from aicore.tracing import InMemoryTracer

PROMPT = "In one sentence, what is a vector database?"


async def main(n: int) -> None:
    """Run ``n`` calls and print their records."""
    settings = LLMSettings()
    provider = build_provider(settings)
    tracer = InMemoryTracer()
    client = LLMClient(provider, settings, tracer=tracer)
    request = LLMRequest(
        messages=[Message(role="user", content=PROMPT)],
        max_tokens=64,
    )

    for _ in range(n):
        await client.complete(request)

    print(f"provider={settings.provider} model={settings.model}")
    for i, rec in enumerate(tracer.records, start=1):
        u = rec.usage
        print(
            f"{i}: {rec.latency_ms:8.0f} ms "
            f"in={u.input_tokens if u else '-'} "
            f"out={u.output_tokens if u else '-'} "
            f"neurons={u.neurons if u else '-'} "
            f"cost_usd={rec.cost_usd}"
        )

    warm = [r.latency_ms for r in tracer.records[1:]]
    if warm:
        print(f"median warm latency: {statistics.median(warm):.0f} ms")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 5))
