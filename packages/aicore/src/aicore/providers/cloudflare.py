"""Adapter for Cloudflare Workers AI.

Calls the native REST (Representational State
Transfer) endpoint ``/ai/run/{model}`` with ``httpx``,
so status codes and Cloudflare's error codes are
visible and can be mapped onto aicore errors.

One adapter serves chat completion, embedding
(``@cf/baai/bge-base-en-v1.5``) and reranking
(``@cf/baai/bge-reranker-base``). All three share
``_run``, so errors are classified identically.
"""

import json

import httpx
from pydantic import BaseModel, ValidationError, field_validator

from aicore.errors import (
    InvalidRequestError,
    LLMError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    QuotaExhaustedError,
)
from aicore.models import (
    EmbedRequest,
    EmbedResponse,
    LLMRequest,
    LLMResponse,
    RerankRequest,
    RerankResponse,
    RerankResult,
    Usage,
)
from aicore.providers._http import map_status_error

_API_ROOT = "https://api.cloudflare.com/client/v4"
# Cloudflare's code for "daily free allocation used up".
_QUOTA_EXHAUSTED = 3036
# BGE's instruction for short queries against passages.
# Documents are embedded without a prefix (asymmetric).
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class _CloudflareUsage(BaseModel):
    """Usage inside ``result.usage``.

    Counts default to 0 because some models omit
    usage entirely. ``neurons`` is optional: it is
    read when Cloudflare sends it, else ``None``.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    neurons: float | None = None


class _ChatResult(BaseModel):
    """The ``result`` object of a text-generation call.

    ``model`` is the model that actually served the
    request. Older models omit it, hence ``None``.
    """

    response: str
    model: str | None = None
    usage: _CloudflareUsage = _CloudflareUsage()

    @field_validator("response", mode="before")
    @classmethod
    def _json_reply_to_text(cls, value: object) -> object:
        """Turn an already-parsed JSON reply back into text.

        When a model (seen with Llama 3.3) replies with pure JSON,
        Workers AI returns ``response`` as an object, not a string.
        Re-serializing keeps ``LLMResponse.text`` a string for every
        reply, so ``complete_structured`` parses it as usual.
        """
        if isinstance(value, dict | list):
            return json.dumps(value)
        return value


class _EmbedResult(BaseModel):
    """The ``result`` object of an embedding call.

    ``data`` holds one vector per input text, in
    input order. ``shape`` is ``[n_texts, dims]``.
    ``usage`` reports input tokens (no neurons).
    """

    shape: list[int]
    data: list[list[float]]
    usage: _CloudflareUsage = _CloudflareUsage()


class _RerankItem(BaseModel):
    """One scored context: ``id`` is its input index."""

    id: int
    score: float


class _RerankResult(BaseModel):
    """The ``result`` object of a rerank call."""

    response: list[_RerankItem]


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

    ``result`` is left untyped here; ``_run``
    validates it against the shape the caller
    expects. It is ``None`` when ``success`` is false.
    """

    result: object = None
    success: bool


class CloudflareProvider:
    """Workers AI provider: chat, embedding and rerank.

    Satisfies ``LLMProvider``, ``EmbeddingProvider``
    and ``RerankProvider``. The provider owns an
    ``httpx.AsyncClient``; call ``aclose`` when done.
    """

    name = "cloudflare"

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
        model = _require_model(request.model)
        result = await self._run(model, _chat_payload(request), _ChatResult)
        return LLMResponse(
            text=result.response,
            model=result.model or model,
            usage=Usage(
                input_tokens=result.usage.prompt_tokens,
                output_tokens=result.usage.completion_tokens,
                neurons=result.usage.neurons,
            ),
        )

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        """Embed ``request.texts`` in one Workers AI call.

        Queries get BGE's retrieval instruction prefix;
        documents are sent as-is. Input tokens come from
        ``result.usage``; output tokens are always 0.

        Raises:
            Same errors as ``complete``.
        """
        model = _require_model(request.model)
        texts = _with_query_prefix(request, model)
        result = await self._run(model, {"text": texts}, _EmbedResult)
        return EmbedResponse(
            vectors=result.data,
            model=model,
            usage=Usage(
                input_tokens=result.usage.prompt_tokens,
                output_tokens=0,
                neurons=result.usage.neurons,
            ),
        )

    async def rerank(self, request: RerankRequest) -> RerankResponse:
        """Score ``request.documents`` against the query.

        Results are re-sorted (highest score first) and
        truncated to ``top_n`` here, so the contract
        holds even if the API's ordering changes.

        Raises:
            Same errors as ``complete``.
        """
        model = _require_model(request.model)
        payload: dict[str, object] = {
            "query": request.query,
            "contexts": [{"text": d} for d in request.documents],
        }
        if request.top_n is not None:
            payload["top_k"] = request.top_n
        result = await self._run(model, payload, _RerankResult)
        ranked = sorted(result.response, key=lambda r: r.score, reverse=True)
        return RerankResponse(
            results=[RerankResult(index=r.id, score=r.score) for r in ranked[: request.top_n]],
            model=model,
            usage=Usage(input_tokens=0, output_tokens=0),
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._http.aclose()

    async def _run[ResultT: BaseModel](
        self,
        model: str,
        payload: dict[str, object],
        result_type: type[ResultT],
    ) -> ResultT:
        """POST ``payload`` to ``model``; return the
        validated ``result`` or raise an aicore error.

        Raises:
            Same errors as ``complete``.
        """
        try:
            response = await self._http.post(model, json=payload)
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
        try:
            return result_type.model_validate(body.result)
        except ValidationError as exc:
            # A shape this adapter does not know; resending will not fix it.
            msg = f"Unexpected Cloudflare result for {model}: {exc.error_count()} error(s)"
            raise InvalidRequestError(msg) from exc


def _require_model(model: str | None) -> str:
    """Return ``model``, or raise if it is unset.

    Cloudflare puts the model in the URL, so there
    is no server-side default to fall back on.
    """
    if model is None:
        msg = "CloudflareProvider needs request.model"
        raise InvalidRequestError(msg)
    return model


def _chat_payload(request: LLMRequest) -> dict[str, object]:
    """Translate an ``LLMRequest`` into the JSON body."""
    return {
        "messages": [m.model_dump() for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }


def _with_query_prefix(request: EmbedRequest, model: str) -> list[str]:
    """Prefix queries with BGE's instruction.

    Only BGE models use this instruction; other
    models and all documents pass through unchanged.
    """
    if request.input_type == "query" and model.startswith("@cf/baai/bge-"):
        return [_BGE_QUERY_PREFIX + t for t in request.texts]
    return list(request.texts)


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
