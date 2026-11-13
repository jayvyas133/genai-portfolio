"""Adapter for a local Ollama server (https://ollama.com).

Talks to Ollama's HTTP API directly with ``httpx`` instead of the ``ollama``
package, so status codes are visible and can be mapped onto aicore errors.
"""

import httpx
from pydantic import BaseModel

from aicore.errors import (
    InvalidRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from aicore.models import LLMRequest, LLMResponse, Usage
from aicore.providers._http import map_status_error


class _OllamaMessage(BaseModel):
    """The ``message`` object inside an Ollama chat response."""

    content: str


class _OllamaChatResponse(BaseModel):
    """The fields we need from ``POST /api/chat`` with ``stream: false``.

    Ollama omits ``prompt_eval_count`` when the prompt was served from its
    cache, so both counts default to 0 instead of failing validation.
    """

    model: str
    message: _OllamaMessage
    prompt_eval_count: int = 0
    eval_count: int = 0


class OllamaProvider:
    """``LLMProvider`` backed by Ollama's ``/api/chat`` endpoint.

    The provider owns an ``httpx.AsyncClient``; call ``aclose`` when done.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Create a provider for the Ollama server at ``base_url``.

        Args:
            base_url: Root URL of the Ollama server.
            transport: Optional ``httpx`` transport. Tests pass an
                ``httpx.MockTransport`` so no real server is needed.
        """
        # timeout=None: LLMClient already applies a per-attempt timeout,
        # and httpx's 5 s default is too short for a model's first load.
        self._http = httpx.AsyncClient(base_url=base_url, timeout=None, transport=transport)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send ``request`` to Ollama and normalize the reply.

        Raises:
            InvalidRequestError: If ``request.model`` is not set, or
                Ollama rejects the request (e.g. unknown model, 404).
            AuthenticationError: On HTTP 401/403 (e.g. behind a proxy).
            RateLimitError: On HTTP 429.
            ProviderUnavailableError: If Ollama is unreachable or 5xx.
            ProviderTimeoutError: If the HTTP call times out.
        """
        if request.model is None:
            msg = "OllamaProvider needs request.model to be set"
            raise InvalidRequestError(msg)

        payload = _to_payload(request)
        try:
            response = await self._http.post("/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            msg = f"Ollama timed out: {exc!r}"
            raise ProviderTimeoutError(msg) from exc
        except httpx.TransportError as exc:
            msg = f"Cannot reach Ollama: {exc!r}"
            raise ProviderUnavailableError(msg) from exc

        if response.is_error:
            raise map_status_error(
                response,
                vendor="Ollama",
                detail=_error_detail(response),
            )

        body = _OllamaChatResponse.model_validate_json(response.content)
        return LLMResponse(
            text=body.message.content,
            model=body.model,
            usage=Usage(
                input_tokens=body.prompt_eval_count,
                output_tokens=body.eval_count,
            ),
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._http.aclose()


def _to_payload(request: LLMRequest) -> dict[str, object]:
    """Translate an ``LLMRequest`` into Ollama's ``/api/chat`` JSON body."""
    return {
        "model": request.model,
        "messages": [m.model_dump() for m in request.messages],
        "stream": False,
        "options": {
            "temperature": request.temperature,
            "num_predict": request.max_tokens,
        },
    }


def _error_detail(response: httpx.Response) -> str:
    """Return Ollama's ``{"error": "..."}`` text, or the raw body."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict) and "error" in body:
        return str(body["error"])
    return response.text[:200]
