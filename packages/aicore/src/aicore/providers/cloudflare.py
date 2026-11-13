"""Adapter for Cloudflare Workers AI.

Calls the native REST (Representational State
Transfer) endpoint ``/ai/run/{model}`` with ``httpx``,
so status codes and Cloudflare's error codes are
visible and can be mapped onto aicore errors.
"""

import httpx
from pydantic import BaseModel, ValidationError

from aicore.errors import (
    InvalidRequestError,
    LLMError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    QuotaExhaustedError,
)
from aicore.models import LLMRequest, LLMResponse, Usage
from aicore.providers._http import map_status_error

_API_ROOT = "https://api.cloudflare.com/client/v4"
# Cloudflare's code for "daily free allocation used up".
_QUOTA_EXHAUSTED = 3036


class _CloudflareUsage(BaseModel):
    """Token counts inside ``result.usage``.

    Counts default to 0 because some models omit
    usage entirely.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0


class _CloudflareResult(BaseModel):
    """The ``result`` object of a text-generation call.

    ``model`` is the model that actually served the
    request. Older models omit it, hence ``None``.
    """

    response: str
    model: str | None = None
    usage: _CloudflareUsage = _CloudflareUsage()


class _CloudflareError(BaseModel):
    """One entry of the envelope's ``errors`` list."""

    code: int
    message: str


class _CloudflareErrorBody(BaseModel):
    """Just the ``errors`` part of any Cloudflare body.

    Used to read error responses, where ``result``
    is ``null`` or missing.
    """

    errors: list[_CloudflareError] = []


class _CloudflareEnvelope(_CloudflareErrorBody):
    """Cloudflare's standard API (Application
    Programming Interface) response wrapper.

    ``result`` is ``None`` when ``success`` is false.
    """

    result: _CloudflareResult | None = None
    success: bool


class CloudflareProvider:
    """``LLMProvider`` backed by Workers AI ``/ai/run``.

    The provider owns an ``httpx.AsyncClient``; call
    ``aclose`` when done.
    """

    def __init__(
        self,
        account_id: str,
        api_token: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Create a provider for one Cloudflare account.

        Args:
            account_id: Cloudflare account ID (in the URL).
            api_token: API token with Workers AI read
                permission. Sent as a Bearer token.
            transport: Optional ``httpx`` transport. Tests
                pass an ``httpx.MockTransport``.
        """
        base_url = f"{_API_ROOT}/accounts/{account_id}/ai/run/"
        # timeout=None: LLMClient owns per-attempt timeouts.
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=None,
            transport=transport,
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send ``request`` to Workers AI and normalize it.

        Raises:
            InvalidRequestError: If ``request.model`` is
                not set, or Cloudflare rejects the request.
            AuthenticationError: On HTTP 401/403.
            RateLimitError: On HTTP 429.
            ProviderUnavailableError: If unreachable or 5xx.
            ProviderTimeoutError: If the HTTP call times out.
            QuotaExhaustedError: If the daily allowance
                is used up (Cloudflare code 3036).
        """
        if request.model is None:
            msg = "CloudflareProvider needs request.model"
            raise InvalidRequestError(msg)

        payload = _to_payload(request)
        try:
            response = await self._http.post(request.model, json=payload)
        except httpx.TimeoutException as exc:
            msg = f"Cloudflare timed out: {exc!r}"
            raise ProviderTimeoutError(msg) from exc
        except httpx.TransportError as exc:
            msg = f"Cannot reach Cloudflare: {exc!r}"
            raise ProviderUnavailableError(msg) from exc

        if response.is_error:
            raise _map_error(response)

        body = _CloudflareEnvelope.model_validate_json(response.content)
        if not body.success or body.result is None:
            detail = _format_errors(body.errors, response)
            msg = f"Cloudflare reported failure: {detail}"
            raise InvalidRequestError(msg)
        result = body.result

        return LLMResponse(
            text=result.response,
            model=result.model or request.model,
            usage=Usage(
                input_tokens=result.usage.prompt_tokens,
                output_tokens=result.usage.completion_tokens,
            ),
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._http.aclose()


def _to_payload(request: LLMRequest) -> dict[str, object]:
    """Translate an ``LLMRequest`` into the JSON body."""
    return {
        "messages": [m.model_dump() for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }


def _parse_errors(
    response: httpx.Response,
) -> list[_CloudflareError]:
    """Return the ``errors`` list, or ``[]`` if the
    body is not Cloudflare's JSON (e.g. a proxy page)."""
    try:
        body = _CloudflareErrorBody.model_validate_json(response.content)
    except ValidationError:
        return []
    return body.errors


def _format_errors(
    errors: list[_CloudflareError],
    response: httpx.Response,
) -> str:
    """Join errors as ``[code] message``, falling back
    to the start of the raw body."""
    if not errors:
        return response.text[:200]
    return "; ".join(f"[{e.code}] {e.message}" for e in errors)


def _map_error(response: httpx.Response) -> LLMError:
    """Map a non-2xx response onto an aicore error.

    Cloudflare's error code wins over the status code
    where they disagree on retryability (3036 arrives
    as HTTP 429 but must not be retried).
    """
    errors = _parse_errors(response)
    detail = _format_errors(errors, response)
    if any(e.code == _QUOTA_EXHAUSTED for e in errors):
        msg = f"Cloudflare daily quota used up: {detail}"
        return QuotaExhaustedError(msg)
    return map_status_error(response, vendor="Cloudflare", detail=detail)
