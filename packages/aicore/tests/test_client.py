"""Unit tests for ``LLMClient.complete``: retries, permanent errors, timeouts, defaults."""

import asyncio

import pytest
from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.errors import (
    AuthenticationError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from aicore.models import LLMRequest, LLMResponse, Message
from aicore.providers.base import LLMProvider
from aicore.providers.fakeai import FakeProvider


def make_request(model: str | None = None) -> LLMRequest:
    """Build a minimal one-message request."""
    return LLMRequest(messages=[Message(role="user", content="hello")], model=model)


def make_client(
    provider: LLMProvider,
    *,
    max_retries: int = 3,
    timeout_s: float = 30.0,
    model: str = "settings-model",
) -> LLMClient:
    """Build a client with zero backoff so retry tests run instantly."""
    settings = LLMSettings(max_retries=max_retries, timeout_s=timeout_s, model=model)
    return LLMClient(provider, settings, base_delay_s=0)


class SlowProvider:
    """Provider that never answers in time, used to exercise the timeout path."""

    name = "slow"

    def __init__(self) -> None:
        """Start with zero recorded calls."""
        self.calls = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Count the call, then sleep far longer than any test timeout."""
        self.calls += 1
        await asyncio.sleep(10)
        raise AssertionError("unreachable: the client should have timed out")


async def test_returns_reply_on_first_success() -> None:
    """A healthy provider is called exactly once."""
    provider = FakeProvider(script=["hi there"])

    response = await make_client(provider).complete(make_request())

    assert response.text == "hi there"
    assert len(provider.calls) == 1


async def test_retries_transient_errors_then_succeeds() -> None:
    """Retryable errors are retried until the provider recovers."""
    provider = FakeProvider(
        script=[ProviderUnavailableError("503"), RateLimitError("429"), "recovered"]
    )

    response = await make_client(provider).complete(make_request())

    assert response.text == "recovered"
    assert len(provider.calls) == 3


async def test_gives_up_after_max_retries() -> None:
    """With max_retries=2 the client makes 3 attempts, then re-raises the last error."""
    provider = FakeProvider(script=[ProviderUnavailableError("503")] * 3)

    with pytest.raises(ProviderUnavailableError):
        await make_client(provider, max_retries=2).complete(make_request())

    assert len(provider.calls) == 3


async def test_non_retryable_error_is_raised_immediately() -> None:
    """A permanent error is never retried."""
    provider = FakeProvider(script=[AuthenticationError("bad key"), "never returned"])

    with pytest.raises(AuthenticationError):
        await make_client(provider).complete(make_request())

    assert len(provider.calls) == 1


async def test_retry_after_overrides_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider-supplied retry_after_s is used as the exact sleep time."""
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        """Stand-in for asyncio.sleep that records the delay instead of waiting."""
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", record_sleep)
    provider = FakeProvider(script=[RateLimitError("429", retry_after_s=7.0), "ok"])

    await make_client(provider).complete(make_request())

    assert delays == [7.0]


async def test_slow_provider_times_out_on_every_attempt() -> None:
    """Each attempt gets its own timeout; after the last one we see ProviderTimeoutError."""
    provider = SlowProvider()

    with pytest.raises(ProviderTimeoutError):
        await make_client(provider, max_retries=1, timeout_s=0.01).complete(make_request())

    assert provider.calls == 2


async def test_model_defaults_to_settings_but_explicit_model_wins() -> None:
    """A request without a model gets the settings model; an explicit one is kept."""
    provider = FakeProvider(script=["a", "b"])
    client = make_client(provider)

    await client.complete(make_request())
    await client.complete(make_request(model="explicit-model"))

    assert provider.calls[0].model == "settings-model"
    assert provider.calls[1].model == "explicit-model"
