"""Tests for how ``LLMClient`` runs guardrails around the cache and provider."""

import pytest
from aicore.cache import InMemoryCache
from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.errors import GuardrailError
from aicore.guardrails import (
    InputGuardrail,
    MaxInputChars,
    OutputGuardrail,
    PIIRedactor,
    SecretLeakGuard,
)
from aicore.models import LLMRequest, Message
from aicore.providers.fakeai import FakeProvider
from aicore.tracing import InMemoryTracer

# Built at runtime so no full token literal is committed.
FAKE_KEY = "AKIA" + "Z" * 16


def make_request(text: str) -> LLMRequest:
    """Build a one-message user request."""
    return LLMRequest(messages=[Message(role="user", content=text)])


def make_client(
    provider: FakeProvider,
    *,
    cache: InMemoryCache | None = None,
    input_guards: tuple[InputGuardrail, ...] = (),
    output_guards: tuple[OutputGuardrail, ...] = (),
) -> tuple[LLMClient, InMemoryTracer]:
    """Return a client with the given guards, and its tracer."""
    tracer = InMemoryTracer()
    client = LLMClient(
        provider,
        LLMSettings(model="m"),
        base_delay_s=0,
        tracer=tracer,
        cache=cache,
        input_guards=input_guards,
        output_guards=output_guards,
    )
    return client, tracer


async def test_input_block_skips_provider_and_is_traced() -> None:
    """A blocked request never reaches the provider, but is recorded."""
    provider = FakeProvider(script=["never sent"])
    client, tracer = make_client(provider, input_guards=(MaxInputChars(5),))

    with pytest.raises(GuardrailError, match="max_input_chars"):
        await client.complete(make_request("far too long"))

    assert provider.calls == []
    [rec] = tracer.records
    assert rec.outcome == "GuardrailError"
    assert rec.attempts == 0
    assert rec.usage is None


async def test_provider_sees_only_redacted_text() -> None:
    """PII is replaced before the request leaves the process."""
    provider = FakeProvider(script=["ok"])
    client, _ = make_client(provider, input_guards=(PIIRedactor(),))

    await client.complete(make_request("I am jay@example.com"))

    [sent] = provider.calls
    assert sent.messages[-1].content == "I am [EMAIL]"


async def test_cache_key_is_built_from_redacted_request() -> None:
    """Same question, different emails: one provider call, then a hit."""
    provider = FakeProvider(script=["answer"])
    client, tracer = make_client(provider, cache=InMemoryCache(), input_guards=(PIIRedactor(),))

    await client.complete(make_request("Reset password for a@x.com"))
    second = await client.complete(make_request("Reset password for b@y.org"))

    assert second.text == "answer"
    assert len(provider.calls) == 1
    assert [r.cached for r in tracer.records] == [False, True]


async def test_output_block_is_not_cached_but_cost_is_traced() -> None:
    """A rejected reply is never stored, and its usage is still recorded."""
    provider = FakeProvider(script=[f"key {FAKE_KEY}", f"key {FAKE_KEY}"])
    client, tracer = make_client(
        provider, cache=InMemoryCache(), output_guards=(SecretLeakGuard(),)
    )

    for _ in range(2):
        with pytest.raises(GuardrailError, match="aws_access_key"):
            await client.complete(make_request("give me a key"))

    assert len(provider.calls) == 2  # nothing was cached
    for rec in tracer.records:
        assert rec.outcome == "GuardrailError"
        assert rec.attempts == 1  # blocked, not retried
        assert rec.usage is not None  # the provider was still paid


async def test_output_guards_also_check_cache_hits() -> None:
    """A bad reply cached before the guard existed is still blocked."""
    cache = InMemoryCache()
    unguarded, _ = make_client(FakeProvider(script=[f"key {FAKE_KEY}"]), cache=cache)
    await unguarded.complete(make_request("q"))

    provider = FakeProvider()
    guarded, tracer = make_client(provider, cache=cache, output_guards=(SecretLeakGuard(),))

    with pytest.raises(GuardrailError):
        await guarded.complete(make_request("q"))

    assert provider.calls == []  # served from cache, then blocked
    [rec] = tracer.records
    assert rec.cached is True
    assert rec.outcome == "GuardrailError"
