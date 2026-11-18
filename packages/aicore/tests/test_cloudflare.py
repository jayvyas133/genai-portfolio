"""Tests for ``CloudflareProvider``.

An ``httpx.MockTransport`` stands in for the Workers
AI API, so unit tests need no network or account.
One real-API test is marked ``integration`` and is
skipped without credentials.
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
    QuotaExhaustedError,
    RateLimitError,
)
from aicore.models import LLMRequest, Message
from aicore.providers.cloudflare import CloudflareProvider

Handler = Callable[[httpx.Request], httpx.Response]

ACCOUNT = "acct-123"
TOKEN = "test-token"
MODEL = "@cf/meta/llama-3.1-8b-instruct"
SERVED = "@cf/meta/llama-3.1-8b-fast-v2"
OK_BODY = {
    "result": {
        "response": "Hello!",
        "model": SERVED,
        "usage": {
            "prompt_tokens": 41,
            "completion_tokens": 7,
            "neurons": 0.95,
        },
    },
    "success": True,
    "errors": [],
    "messages": [],
}


def error_body(code: int, message: str) -> dict[str, object]:
    """Return Cloudflare's error envelope for one error."""
    return {
        "result": None,
        "success": False,
        "errors": [{"code": code, "message": message}],
        "messages": [],
    }


def make_provider(handler: Handler) -> CloudflareProvider:
    """Return a provider answered by ``handler``."""
    transport = httpx.MockTransport(handler)
    return CloudflareProvider(ACCOUNT, TOKEN, transport=transport)


def make_request(model: str | None = MODEL) -> LLMRequest:
    """Return a one-message request for ``model``."""
    msg = Message(role="user", content="hi")
    return LLMRequest(messages=[msg], model=model)


def recording(
    *responses: httpx.Response,
) -> tuple[Handler, list[httpx.Request]]:
    """Return a handler replaying ``responses`` and
    the list of requests it saw."""
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the request; answer with the next one."""
        seen.append(request)
        return queue.pop(0)

    return handler, seen


def make_client(provider: CloudflareProvider) -> LLMClient:
    """Return a client with retries on and no sleep."""
    settings = LLMSettings(model=MODEL, max_retries=3)
    return LLMClient(provider, settings, base_delay_s=0)


async def test_complete_maps_reply_and_usage() -> None:
    """A 200 reply becomes an ``LLMResponse``, reporting
    the model that actually served the request."""
    provider = make_provider(lambda r: httpx.Response(200, json=OK_BODY))

    reply = await provider.complete(make_request())

    assert reply.text == "Hello!"
    assert reply.model == SERVED
    assert reply.usage.input_tokens == 41
    assert reply.usage.output_tokens == 7
    assert reply.usage.neurons == 0.95


async def test_complete_turns_a_parsed_json_reply_back_into_text() -> None:
    """Workers AI returns pure-JSON replies as an object; the text is that JSON."""
    body = {**OK_BODY, "result": {"response": {"answer": "x", "refused": False}}}
    provider = make_provider(lambda r: httpx.Response(200, json=body))

    reply = await provider.complete(make_request())

    assert json.loads(reply.text) == {"answer": "x", "refused": False}


async def test_complete_rejects_an_unknown_result_shape() -> None:
    """A result the adapter cannot read is an aicore error, not a pydantic one."""
    body = {**OK_BODY, "result": {"response": 42}}
    provider = make_provider(lambda r: httpx.Response(200, json=body))

    with pytest.raises(InvalidRequestError, match="Unexpected Cloudflare result"):
        await provider.complete(make_request())


async def test_complete_sends_expected_request() -> None:
    """URL holds account and model; token is a Bearer
    header; the body has no ``model`` key."""
    handler, seen = recording(httpx.Response(200, json=OK_BODY))

    await make_provider(handler).complete(make_request())

    sent = seen[0]
    assert sent.method == "POST"
    assert sent.url.path == (f"/client/v4/accounts/{ACCOUNT}/ai/run/{MODEL}")
    assert sent.headers["Authorization"] == f"Bearer {TOKEN}"
    assert json.loads(sent.content) == {
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
        "max_tokens": 1024,
    }


async def test_missing_model_and_usage_fall_back() -> None:
    """Older models omit ``model`` and ``usage``; we
    report the requested model and 0 tokens."""
    body = {"result": {"response": "Hi"}, "success": True}
    provider = make_provider(lambda r: httpx.Response(200, json=body))

    reply = await provider.complete(make_request())

    assert reply.model == MODEL
    assert reply.usage.input_tokens == 0
    assert reply.usage.output_tokens == 0
    assert reply.usage.neurons is None


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, InvalidRequestError),
        (401, AuthenticationError),
        (403, AuthenticationError),
        (429, RateLimitError),
        (500, ProviderUnavailableError),
    ],
)
async def test_http_status_maps_to_aicore_error(status: int, expected: type[LLMError]) -> None:
    """Codes without a Cloudflare override use the
    shared status mapping."""
    body = error_body(1000, "something failed")
    provider = make_provider(lambda r: httpx.Response(status, json=body))

    with pytest.raises(expected, match=r"\[1000\]"):
        await provider.complete(make_request())


async def test_quota_exhausted_is_not_retried() -> None:
    """429 + code 3036 means the daily allowance is
    gone: one call, then ``QuotaExhaustedError``."""
    body = error_body(3036, "daily free allocation used")
    handler, seen = recording(httpx.Response(429, json=body))
    client = make_client(make_provider(handler))

    with pytest.raises(QuotaExhaustedError):
        await client.complete(make_request())

    assert len(seen) == 1


async def test_capacity_limit_is_retried() -> None:
    """429 + code 3040 is temporary, so ``LLMClient``
    retries and then succeeds."""
    handler, seen = recording(
        httpx.Response(429, json=error_body(3040, "busy")),
        httpx.Response(200, json=OK_BODY),
    )
    client = make_client(make_provider(handler))

    reply = await client.complete(make_request())

    assert reply.text == "Hello!"
    assert len(seen) == 2


async def test_success_false_on_200_is_an_error() -> None:
    """The envelope can report failure with HTTP 200."""
    body = error_body(5006, "bad input")
    provider = make_provider(lambda r: httpx.Response(200, json=body))

    with pytest.raises(InvalidRequestError, match="bad input"):
        await provider.complete(make_request())


async def test_non_json_error_body_uses_raw_text() -> None:
    """A proxy's HTML error page must not crash the
    error handler; its text is used instead."""
    provider = make_provider(lambda r: httpx.Response(502, text="<h1>Bad gateway</h1>"))

    with pytest.raises(ProviderUnavailableError, match="Bad gateway"):
        await provider.complete(make_request())


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
    """Network failures are mapped and keep the cause."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Fail the way a real network call would."""
        raise error

    with pytest.raises(expected) as exc_info:
        await make_provider(handler).complete(make_request())

    assert exc_info.value.__cause__ is error


async def test_missing_model_is_rejected_before_http() -> None:
    """Without a model we fail fast, no network call."""
    handler, seen = recording(httpx.Response(200, json=OK_BODY))

    with pytest.raises(InvalidRequestError):
        await make_provider(handler).complete(make_request(model=None))

    assert seen == []


@pytest.mark.integration
async def test_real_cloudflare_round_trip() -> None:
    """Call the real Workers AI API (uses a few neurons).

    Skipped unless ``AICORE_API_KEY`` and
    ``AICORE_CLOUDFLARE_ACCOUNT_ID`` are set.
    """
    settings = LLMSettings(model=MODEL, max_retries=0)
    if settings.api_key is None or not settings.cloudflare_account_id:
        pytest.skip("Cloudflare credentials not set")

    provider = CloudflareProvider(
        settings.cloudflare_account_id,
        settings.api_key.get_secret_value(),
    )
    client = LLMClient(provider, settings)
    msg = Message(role="user", content="Reply with: pong")
    try:
        reply = await client.complete(LLMRequest(messages=[msg], max_tokens=20))
    finally:
        await provider.aclose()

    assert reply.text.strip()
    assert reply.usage.output_tokens > 0
