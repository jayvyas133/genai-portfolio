"""The single entry point for LLM calls in every project.

``LLMClient`` wraps any ``LLMProvider`` and adds per-attempt timeouts and
retries with exponential backoff and jitter, so individual adapters stay small
and reliability behaves the same across all backends.
"""

import asyncio
import json
import logging
import random
import time
from datetime import UTC, datetime

from pydantic import BaseModel, ValidationError

from aicore.cache import ResponseCache, cache_key
from aicore.config import LLMSettings
from aicore.errors import (
    ProviderTimeoutError,
    RateLimitError,
    RetryableError,
    StructuredOutputError,
)
from aicore.models import LLMRequest, LLMResponse, Message, Usage
from aicore.pricing import cost_usd
from aicore.providers.base import LLMProvider
from aicore.tracing import CallRecord, NullTracer, Tracer

_log = logging.getLogger(__name__)


class LLMClient:
    """Provider-agnostic LLM client with timeouts and retries.

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
        """
        self._provider = provider
        self._settings = settings or LLMSettings()
        self._base_delay_s = base_delay_s
        self._max_delay_s = max_delay_s
        self._tracer = tracer or NullTracer()
        self._cache = cache

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send ``request``, retrying transient failures with backoff.

        Makes at most ``1 + settings.max_retries`` attempts.
        Sends exactly one ``CallRecord`` to the tracer,
        whether the call succeeds or fails.
        With a cache and ``temperature == 0``, a stored reply
        is returned without calling the provider, and every
        success is stored. Cache errors are logged, never raised.

        Raises:
            RetryableError: If every attempt failed with a transient error.
            NonRetryableError: On the first permanent failure.
        """
        if request.model is None:
            request = request.model_copy(update={"model": self._settings.model})

        key = self._cache_key_for(request)
        started_at = datetime.now(UTC)
        start = time.perf_counter()
        attempts = 0
        response: LLMResponse | None = None
        outcome = "ok"
        cached = False
        try:
            if key is not None:
                response = await self._cache_get(key)
                if response is not None:
                    cached = True
                    return response
            while True:
                attempts += 1
                try:
                    response = await self._call_once(request)
                    if key is not None:
                        await self._cache_set(key, response)
                    return response
                except RetryableError as err:
                    if attempts > self._settings.max_retries:
                        raise  # out of retries: surface the last error
                    delay = self._backoff_delay(attempts - 1, err)
                    await asyncio.sleep(delay)
        except BaseException as err:
            outcome = type(err).__name__
            raise
        finally:
            self._trace(
                request,
                response,
                started_at,
                start,
                attempts,
                outcome,
                cached,
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

    async def _call_once(self, request: LLMRequest) -> LLMResponse:
        """Make one provider call, converting a timeout into ``ProviderTimeoutError``."""
        timeout_s = self._settings.timeout_s
        try:
            return await asyncio.wait_for(self._provider.complete(request), timeout=timeout_s)
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
        request: LLMRequest,
        response: LLMResponse | None,
        started_at: datetime,
        start: float,
        attempts: int,
        outcome: str,
        cached: bool,
    ) -> None:
        """Build a ``CallRecord`` and hand it to the tracer.

        Any error here is logged and swallowed:
        tracing must never break or change the
        result of an LLM call.
        """
        try:
            latency_ms = (time.perf_counter() - start) * 1000
            provider = self._provider.name
            usage = response.usage if response else None
            model = response.model if response else request.model
            rec = CallRecord(
                provider=provider,
                model=model or "unknown",
                started_at=started_at,
                latency_ms=latency_ms,
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
