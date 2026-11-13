"""Tests for ``OllamaProvider``.

An ``httpx.MockTransport`` stands in for the Ollama server, so the unit
tests need no network. One real-server test is marked ``integration``.
"""

import json
from collections.abc import Callable

import httpx
import pytest
from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.errors import (
    AuthenticationError,
    InvalidRequestError,
    LLMError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from aicore.models import LLMRequest, Message
from aicore.providers.ollama import OllamaProvider

Handler = Callable[[httpx.Request], httpx.Response]

MODEL = "qwen2.5:0.5b"
OK_BODY = {
    "model": MODEL,
    "message": {"role": "assistant", "content": "Hello!"},
    "prompt_eval_count": 7,
    "eval_count": 3,
}


def make_provider(handler: Handler) -> OllamaProvider:
    """Return a provider whose HTTP calls are answered by ``handler``."""
    transport = httpx.MockTransport(handler)
    return OllamaProvider("http://ollama.test", transport=transport)


def make_request(model: str | None = MODEL) -> LLMRequest:
    """Return a one-message request for ``model``."""
    msg = Message(role="user", content="hi")
    return LLMRequest(messages=[msg], model=model)


def recording(
    *responses: httpx.Response,
) -> tuple[Handler, list[httpx.Request]]:
    """Return a handler replaying ``responses`` and the requests it saw."""
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the request and answer with the next response."""
        seen.append(request)
        return queue.pop(0)

    return handler, seen


def make_client(provider: OllamaProvider) -> LLMClient:
    """Return a client with retries on and no backoff sleep."""
    settings = LLMSettings(model=MODEL, max_retries=3)
    return LLMClient(provider, settings, base_delay_s=0)


async def test_complete_maps_reply_and_usage() -> None:
    """A 200 reply becomes an ``LLMResponse`` with token counts."""
    provider = make_provider(lambda r: httpx.Response(200, json=OK_BODY))

    reply = await provider.complete(make_request())

    assert reply.text == "Hello!"
    assert reply.model == MODEL
    assert reply.usage.input_tokens == 7
    assert reply.usage.output_tokens == 3


async def test_complete_sends_expected_payload() -> None:
    """The request is translated into Ollama's ``/api/chat`` body."""
    handler, seen = recording(httpx.Response(200, json=OK_BODY))

    await make_provider(handler).complete(make_request())

    assert seen[0].method == "POST"
    assert seen[0].url.path == "/api/chat"
    assert json.loads(seen[0].content) == {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 1024},
    }


async def test_missing_token_counts_default_to_zero() -> None:
    """Ollama may omit counts (e.g. cached prompt); we report 0."""
    body = {"model": MODEL, "message": {"content": "Hi"}}
    provider = make_provider(lambda r: httpx.Response(200, json=body))

    reply = await provider.complete(make_request())

    assert reply.usage.input_tokens == 0
    assert reply.usage.output_tokens == 0


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, InvalidRequestError),
        (401, AuthenticationError),
        (403, AuthenticationError),
        (404, InvalidRequestError),
        (429, RateLimitError),
        (500, ProviderUnavailableError),
        (503, ProviderUnavailableError),
    ],
)
async def test_http_status_maps_to_aicore_error(status: int, expected: type[LLMError]) -> None:
    """Each error status becomes the matching aicore error."""
    provider = make_provider(lambda r: httpx.Response(status))

    with pytest.raises(expected):
        await provider.complete(make_request())


async def test_error_message_includes_ollama_detail() -> None:
    """Ollama's ``{"error": ...}`` text ends up in the exception."""
    body = {"error": "model 'nope' not found"}
    provider = make_provider(lambda r: httpx.Response(404, json=body))

    with pytest.raises(InvalidRequestError, match="model 'nope' not found"):
        await provider.complete(make_request())


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"Retry-After": "7"}, 7.0),
        ({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, None),
        ({}, None),
    ],
)
async def test_rate_limit_reads_retry_after(
    headers: dict[str, str], expected: float | None
) -> None:
    """Seconds are parsed; a date or a missing header gives ``None``."""
    provider = make_provider(lambda r: httpx.Response(429, headers=headers))

    with pytest.raises(RateLimitError) as exc_info:
        await provider.complete(make_request())

    assert exc_info.value.retry_after_s == expected


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (httpx.ConnectError("refused"), ProviderUnavailableError),
        (httpx.ReadTimeout("slow"), ProviderTimeoutError),
    ],
)
async def test_transport_error_maps_to_aicore_error(
    error: httpx.TransportError, expected: type[LLMError]
) -> None:
    """Network failures are mapped and keep the original as the cause."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Fail the way a real network call would."""
        raise error

    with pytest.raises(expected) as exc_info:
        await make_provider(handler).complete(make_request())

    assert exc_info.value.__cause__ is error


async def test_missing_model_is_rejected_before_http() -> None:
    """Without a model we fail fast and never touch the network."""
    handler, seen = recording(httpx.Response(200, json=OK_BODY))

    with pytest.raises(InvalidRequestError):
        await make_provider(handler).complete(make_request(model=None))

    assert seen == []


async def test_client_retries_server_error_then_succeeds() -> None:
    """A 503 is retryable, so ``LLMClient`` tries again and succeeds."""
    handler, seen = recording(
        httpx.Response(503),
        httpx.Response(200, json=OK_BODY),
    )
    client = make_client(make_provider(handler))

    reply = await client.complete(make_request())

    assert reply.text == "Hello!"
    assert len(seen) == 2


async def test_client_does_not_retry_unknown_model() -> None:
    """A 404 is permanent, so ``LLMClient`` gives up after one call."""
    handler, seen = recording(
        httpx.Response(404, json={"error": "model not found"}),
    )
    client = make_client(make_provider(handler))

    with pytest.raises(InvalidRequestError):
        await client.complete(make_request())

    assert len(seen) == 1


@pytest.mark.integration
async def test_real_ollama_round_trip() -> None:
    """Talk to a real local Ollama (needs ``ollama pull qwen2.5:0.5b``).

    The long timeout covers the first call, which loads the model into
    RAM (Random Access Memory). The reply text is not asserted because
    model output is not deterministic.
    """
    settings = LLMSettings(provider="ollama", model=MODEL, timeout_s=180, max_retries=0)
    provider = OllamaProvider(str(settings.ollama_base_url))
    client = LLMClient(provider, settings)
    msg = Message(role="user", content="Reply with one word: pong")
    try:
        reply = await client.complete(LLMRequest(messages=[msg], max_tokens=20))
    finally:
        await provider.aclose()

    assert reply.text.strip()
    assert reply.usage.output_tokens > 0
