"""The single entry point for model calls in every project.

``LLMClient`` wraps any ``LLMProvider`` and adds guardrails, caching,
per-attempt timeouts and retries with exponential backoff and jitter,
and tracing, so individual adapters stay small and behave the same
across all backends. Embedding and reranking share the same retry,
timeout and tracing path through optional ``embedder`` and
``reranker`` providers.
"""

import asyncio
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, ValidationError

from aicore.cache import ResponseCache, cache_key
from aicore.config import LLMSettings
from aicore.errors import (
    InvalidRequestError,
    ProviderTimeoutError,
    RateLimitError,
    RetryableError,
    StructuredOutputError,
)
from aicore.guardrails import (
    InputGuardrail,
    OutputGuardrail,
    apply_input_guards,
    apply_output_guards,
)
from aicore.models import (
    EmbedRequest,
    EmbedResponse,
    LLMRequest,
    LLMResponse,
    Message,
    RerankRequest,
    RerankResponse,
    Usage,
)
from aicore.pricing import cost_usd
from aicore.providers.base import EmbeddingProvider, LLMProvider, RerankProvider
from aicore.tracing import CallRecord, NullTracer, Operation, Tracer

_log = logging.getLogger(__name__)


@dataclass
class _Attempts:
    """Attempt counter shared with the retry loop.

    Mutable so the caller can still read the count
    (for tracing) after the loop raises.
    """

    count: int = 0


class LLMClient:
    """Provider-agnostic model client with timeouts and retries.

    Only ``RetryableError`` failures are retried. Any other error (for example
    ``AuthenticationError``) is raised immediately, because sending the same
    request again cannot fix it.
    """

    def __init__(
        self,
        provider: LLMProvider,
        settings: LLMSettings | None = None,
        *,
        base_delay_s: float = 0.5,
        max_delay_s: float = 8.0,
        tracer: Tracer | None = None,
        cache: ResponseCache | None = None,
        input_guards: Sequence[InputGuardrail] = (),
        output_guards: Sequence[OutputGuardrail] = (),
        embedder: EmbeddingProvider | None = None,
        reranker: RerankProvider | None = None,
    ) -> None:
        """Create a client around ``provider``.

        Args:
            provider: Any object satisfying the ``LLMProvider`` protocol.
            settings: Timeout, retry and model defaults; loaded from the
                environment when omitted.
            base_delay_s: Backoff ceiling for the first retry; doubles each time.
                Tests pass 0 so retries happen instantly.
            max_delay_s: Upper bound for the backoff ceiling.
            tracer: Receives one ``CallRecord`` per call.
                Defaults to ``NullTracer`` (no tracing).
            cache: Stores replies to ``temperature == 0``
                requests. Defaults to ``None`` (no caching).
            input_guards: Run in order before the cache and the
                provider; may redact the request or block it.
            output_guards: Run on every reply (including cache hits)
                before it is cached or returned.
            embedder: Backend for ``embed``. Often the same object
                as ``provider``; ``None`` disables ``embed``.
            reranker: Backend for ``rerank``. ``None`` disables
                ``rerank`` (e.g. Ollama has no rerank endpoint).
        """
        self._provider = provider
        self._settings = settings or LLMSettings()
        self._base_delay_s = base_delay_s
        self._max_delay_s = max_delay_s
        self._tracer = tracer or NullTracer()
        self._cache = cache
        self._input_guards = tuple(input_guards)
        self._output_guards = tuple(output_guards)
        self._embedder = embedder
        self._reranker = reranker

    async def aclose(self) -> None:
        """Close every backend that holds a connection pool.

        The same adapter often serves chat, embed and rerank, so each
        distinct backend is closed once. Backends without ``aclose``
        (like ``FakeProvider``) are skipped.
        """
        backends = (self._provider, self._embedder, self._reranker)
        unique = {id(b): b for b in backends if b is not None}
        for backend in unique.values():
            close = getattr(backend, "aclose", None)
            if close is not None:
                await close()

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send ``request``, retrying transient failures with backoff.

        Order: input guards, cache lookup, provider (with
        retries, only on a miss), output guards, cache write.
        Makes at most ``1 + settings.max_retries`` attempts.
        Sends exactly one ``CallRecord`` to the tracer,
        whether the call succeeds, fails or is blocked.
        With a cache and ``temperature == 0``, a stored reply
        is returned without calling the provider. Cache errors
        are logged, never raised.

        Raises:
            GuardrailError: If an input or output guard blocks the call.
            RetryableError: If every attempt failed with a transient error.
            NonRetryableError: On the first permanent failure.
        """
        if request.model is None:
            request = request.model_copy(update={"model": self._settings.model})

        started_at = datetime.now(UTC)
        start = time.perf_counter()
        attempts = _Attempts()
        response: LLMResponse | None = None
        outcome = "ok"
        cached = False
        try:
            # Guard first: the redacted request is what gets
            # hashed, cached, sent and traced.
            request = apply_input_guards(self._input_guards, request)
            key = self._cache_key_for(request)
            if key is not None:
                response = await self._cache_get(key)
                cached = response is not None
            if response is None:
                response = await self._with_retries(
                    lambda: self._provider.complete(request), attempts
                )
            # Also on hits: guards may have changed since the write.
            apply_output_guards(self._output_guards, request, response)
            if key is not None and not cached:
                await self._cache_set(key, response)
            return response
        except BaseException as err:
            outcome = type(err).__name__
            raise
        finally:
            self._trace(
                "complete",
                self._provider.name,
                response.model if response else request.model,
                response.usage if response else None,
                started_at,
                start,
                attempts.count,
                outcome,
                cached,
            )

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        """Embed ``request.texts`` with timeouts, retries and tracing.

        ``request.model`` falls back to ``settings.embed_model``.
        Guards and the response cache do not apply. Sends exactly
        one ``CallRecord`` with ``operation="embed"``.

        Raises:
            InvalidRequestError: If the client has no ``embedder``.
            RetryableError: If every attempt failed with a transient error.
            NonRetryableError: On the first permanent failure.
        """
        embedder = self._embedder
        if embedder is None:
            msg = "LLMClient was created without an embedder"
            raise InvalidRequestError(msg)
        if request.model is None:
            request = request.model_copy(update={"model": self._settings.embed_model})
        return await self._run_traced(
            "embed", embedder.name, request.model, lambda: embedder.embed(request)
        )

    async def rerank(self, request: RerankRequest) -> RerankResponse:
        """Rerank documents with timeouts, retries and tracing.

        ``request.model`` falls back to ``settings.rerank_model``.
        Sends exactly one ``CallRecord`` with ``operation="rerank"``.

        Raises:
            InvalidRequestError: If the client has no ``reranker``.
            RetryableError: If every attempt failed with a transient error.
            NonRetryableError: On the first permanent failure.
        """
        reranker = self._reranker
        if reranker is None:
            msg = "LLMClient was created without a reranker"
            raise InvalidRequestError(msg)
        if request.model is None:
            request = request.model_copy(update={"model": self._settings.rerank_model})
        return await self._run_traced(
            "rerank", reranker.name, request.model, lambda: reranker.rerank(request)
        )

    async def complete_structured[T: BaseModel](self, request: LLMRequest, schema: type[T]) -> T:
        """Send ``request`` and parse the reply into an instance of ``schema``.

        The JSON Schema of ``schema`` is added as a system message so the model
        knows the exact shape to return. Transport failures are retried exactly
        as in ``complete``; a reply that does not match the schema is not.

        Raises:
            StructuredOutputError: If the reply is not valid JSON for ``schema``.
        """

        instruction = Message(
            role="system",
            content=(
                "Reply with only a JSON object that matches this JSON Schema. "
                "Do not add explanations or code fences.\n" + json.dumps(schema.model_json_schema())
            ),
        )
        structured = request.model_copy(update={"messages": [instruction, *request.messages]})
        response = await self.complete(structured)
        try:
            return schema.model_validate_json(_strip_code_fences(response.text))
        except ValidationError as err:
            raise StructuredOutputError(
                f"reply did not match {schema.__name__}: {err.error_count()} error(s)",
                raw_text=response.text,
            ) from err

    async def _run_traced[R: (EmbedResponse, RerankResponse)](
        self,
        operation: Operation,
        provider: str,
        model: str | None,
        call: Callable[[], Awaitable[R]],
    ) -> R:
        """Run ``call`` with retries and send exactly one ``CallRecord``."""
        started_at = datetime.now(UTC)
        start = time.perf_counter()
        attempts = _Attempts()
        response: R | None = None
        outcome = "ok"
        try:
            response = await self._with_retries(call, attempts)
            return response
        except BaseException as err:
            outcome = type(err).__name__
            raise
        finally:
            self._trace(
                operation,
                provider,
                response.model if response else model,
                response.usage if response else None,
                started_at,
                start,
                attempts.count,
                outcome,
                cached=False,
            )

    async def _with_retries[T](self, call: Callable[[], Awaitable[T]], attempts: _Attempts) -> T:
        """Await ``call()`` with a timeout, retrying transient errors.

        ``call`` is a factory, not a coroutine: a coroutine can be
        awaited only once, so each attempt needs a fresh one.
        ``attempts`` is updated in place so the caller can trace
        the count even when every attempt fails.
        """
        while True:
            attempts.count += 1
            try:
                return await self._with_timeout(call())
            except RetryableError as err:
                if attempts.count > self._settings.max_retries:
                    raise  # out of retries: surface the last error
                await asyncio.sleep(self._backoff_delay(attempts.count - 1, err))

    async def _with_timeout[T](self, awaitable: Awaitable[T]) -> T:
        """Await one provider call, converting a timeout into ``ProviderTimeoutError``."""
        timeout_s = self._settings.timeout_s
        try:
            return await asyncio.wait_for(awaitable, timeout=timeout_s)
        except TimeoutError as err:
            raise ProviderTimeoutError(f"provider did not respond within {timeout_s}s") from err

    def _backoff_delay(self, attempt: int, err: RetryableError) -> float:
        """Return how long to sleep before the next attempt.

        A provider-supplied ``retry_after_s`` wins. Otherwise use "full jitter":
        a random delay between 0 and an exponentially growing ceiling.
        """
        if isinstance(err, RateLimitError) and err.retry_after_s is not None:
            return err.retry_after_s
        ceiling = min(self._max_delay_s, self._base_delay_s * 2**attempt)
        return random.uniform(0, ceiling)

    def _cache_key_for(self, request: LLMRequest) -> str | None:
        """Return the cache key, or None if not cacheable.

        Only deterministic requests (``temperature == 0``)
        are cached; a sampled reply is meant to vary.
        """
        if self._cache is None or request.temperature != 0:
            return None
        return cache_key(self._provider.name, request)

    async def _cache_get(self, key: str) -> LLMResponse | None:
        """Look up ``key``; a broken cache counts as a miss."""
        if self._cache is None:
            return None
        try:
            return await self._cache.get(key)
        except Exception:
            _log.warning("cache read failed", exc_info=True)
            return None

    async def _cache_set(self, key: str, response: LLMResponse) -> None:
        """Store ``response``; failures are logged, not raised."""
        if self._cache is None:
            return
        try:
            await self._cache.set(key, response)
        except Exception:
            _log.warning("cache write failed", exc_info=True)

    def _trace(
        self,
        operation: Operation,
        provider: str,
        model: str | None,
        usage: Usage | None,
        started_at: datetime,
        start: float,
        attempts: int,
        outcome: str,
        cached: bool,
    ) -> None:
        """Build a ``CallRecord`` and hand it to the tracer.

        Any error here is logged and swallowed:
        tracing must never break or change the
        result of a model call.
        """
        try:
            rec = CallRecord(
                operation=operation,
                provider=provider,
                model=model or "unknown",
                started_at=started_at,
                latency_ms=(time.perf_counter() - start) * 1000,
                attempts=attempts,
                outcome=outcome,
                cached=cached,
                usage=usage,
                cost_usd=self._cost(provider, usage, cached),
            )
            self._tracer.record(rec)
        except Exception:
            _log.warning("tracing failed", exc_info=True)

    @staticmethod
    def _cost(provider: str, usage: Usage | None, cached: bool) -> float | None:
        """Return the call's cost in USD.

        A cache hit cost nothing; otherwise use the
        price table (``None`` when unknown or failed).
        """
        if cached:
            return 0.0
        return cost_usd(provider, usage) if usage else None


def _strip_code_fences(text: str) -> str:
    """Remove a surrounding Markdown code fence (for example ```json ... ```), if any.

    Models often wrap JSON in fences despite being told not to. Tolerating
    that is cheaper than failing the call and asking again.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        # drop the opening fence line (e.g. "```json"), then the closing fence
        stripped = stripped.partition("\n")[2].removesuffix("```")
    return stripped.strip()
