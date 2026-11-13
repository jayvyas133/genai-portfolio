# ADR (Architecture Decision Record) 0001: Provider interface

- **Status:** accepted
- **Date:** 2026-11-13
- **Scope:** aicore

## Context

All five projects call an LLM (Large Language Model).
We want to try several vendors (Gemini, Ollama,
Cloudflare) and run unit tests with no network
and no API (Application Programming Interface) key.
Retries, timeouts and structured output must work
the same way for every vendor.

## Options considered

1. **Call each vendor SDK (Software Development Kit)
   directly in projects.**
   Pro: no extra layer.
   Con: retry logic copied per project; changing
   vendor touches every call site; tests need mocks
   of each SDK.
2. **Use a framework such as LangChain.**
   Pro: many adapters out of the box.
   Con: large dependency, fast-moving API, hides the
   retry and error behaviour we must explain.
   Banned by the portfolio rules.
3. **Own a small `LLMProvider` protocol plus one
   `LLMClient` that wraps it.**
   Pro: adapters stay tiny; reliability lives in one
   place; a scripted `FakeProvider` makes tests fast
   and offline.
   Con: we write and maintain each adapter ourselves.

## Decision

Option 3. `LLMProvider` is a `typing.Protocol` with a
single `async complete(request) -> LLMResponse`.
A protocol (structural typing) was chosen over an
ABC (Abstract Base Class) so adapters and test doubles
do not need to inherit from aicore classes; mypy still
checks that they match.

`LLMClient` owns the cross-cutting behaviour:

- per-attempt timeout via `asyncio.wait_for`;
- retries only for `RetryableError`, with full-jitter
  exponential backoff;
- `retry_after_s` from the provider beats the
  computed delay;
- structured output via JSON (JavaScript Object
  Notation) Schema plus Pydantic validation.

Adapters must map vendor errors onto aicore errors.
That mapping is what lets the client decide what is
safe to retry without knowing any vendor.

## Trade-offs

- Each new vendor costs one adapter (roughly one small
  file plus its error mapping).
- Only the features in `LLMRequest` are portable.
  Vendor-only features (for example, native JSON mode)
  need new fields or stay unused.
- Schema mismatches are not retried. Revisit once the
  eval harness (M4) shows how often they happen.
- Streaming is not supported yet. Revisit when a
  project needs token-by-token output.
