"""Unit tests for tracing in ``LLMClient``."""

import pytest
from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.errors import (
    AuthenticationError,
    ProviderUnavailableError,
    RateLimitError,
)
from aicore.models import LLMRequest, Message
from aicore.providers.fakeai import FakeProvider, Outcome
from aicore.tracing import CallRecord, InMemoryTracer


def make_request() -> LLMRequest:
    """Build a minimal one-message request."""
    return LLMRequest(messages=[Message(role="user", content="hello")])


def make_client(
    script: list[Outcome],
) -> tuple[LLMClient, InMemoryTracer]:
    """Return a no-sleep client and its tracer."""
    tracer = InMemoryTracer()
    settings = LLMSettings(model="m", max_retries=3)
    client = LLMClient(
        FakeProvider(script=script),
        settings,
        base_delay_s=0,
        tracer=tracer,
    )
    return client, tracer


class BrokenTracer:
    """Tracer that always fails."""

    def record(self, rec: CallRecord) -> None:
        """Raise, to simulate a broken backend."""
        raise RuntimeError("tracer down")


async def test_success_records_one_ok_call() -> None:
    """A first-try success gives one full record."""
    client, tracer = make_client(["hi there"])

    await client.complete(make_request())

    [rec] = tracer.records
    assert rec.provider == "fake"
    assert rec.model == "m"
    assert rec.outcome == "ok"
    assert rec.attempts == 1
    assert rec.usage is not None
    assert rec.usage.output_tokens == 2
    assert rec.cost_usd == 0.0
    assert rec.latency_ms >= 0


async def test_retry_then_success_counts_attempts() -> None:
    """Retries stay in one record, as ``attempts``."""
    client, tracer = make_client([ProviderUnavailableError("down"), "ok now"])

    await client.complete(make_request())

    [rec] = tracer.records
    assert rec.outcome == "ok"
    assert rec.attempts == 2


async def test_final_failure_is_recorded() -> None:
    """Giving up still records, with no usage."""
    client, tracer = make_client([RateLimitError("slow down")] * 4)

    with pytest.raises(RateLimitError):
        await client.complete(make_request())

    [rec] = tracer.records
    assert rec.outcome == "RateLimitError"
    assert rec.attempts == 4
    assert rec.usage is None
    assert rec.cost_usd is None


async def test_permanent_error_recorded_once() -> None:
    """A permanent error: one attempt, one record."""
    client, tracer = make_client([AuthenticationError("bad key")])

    with pytest.raises(AuthenticationError):
        await client.complete(make_request())

    [rec] = tracer.records
    assert rec.outcome == "AuthenticationError"
    assert rec.attempts == 1


async def test_broken_tracer_does_not_break_call() -> None:
    """Tracing errors are logged, not raised."""
    client = LLMClient(
        FakeProvider(script=["still works"]),
        LLMSettings(model="m"),
        tracer=BrokenTracer(),
    )

    reply = await client.complete(make_request())

    assert reply.text == "still works"


def test_record_defaults_to_not_cached() -> None:
    """Existing callers get ``cached=False``."""
    rec = CallRecord(provider="fake", model="m", latency_ms=1, attempts=1, outcome="ok")
    assert rec.cached is False


def test_cache_hit_record_allows_zero_attempts() -> None:
    """A hit made no provider call, so 0 attempts is valid."""
    rec = CallRecord(
        provider="fake",
        model="m",
        latency_ms=0.1,
        attempts=0,
        outcome="ok",
        cached=True,
        cost_usd=0.0,
    )
    assert rec.attempts == 0
    assert rec.cached is True
