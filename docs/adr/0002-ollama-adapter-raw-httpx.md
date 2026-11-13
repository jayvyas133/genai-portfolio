# ADR (Architecture Decision Record) 0002: Ollama adapter with raw httpx

- **Status:** accepted
- **Date:** 2026-11-13
- **Scope:** aicore

## Context

We want a free, local LLM (Large Language Model) for development and
demos, with no API (Application Programming Interface) key and no
per-call cost. Ollama runs small models such as `qwen2.5:0.5b` on a
CPU-only (Central Processing Unit) laptop. Per ADR 0001, the adapter
must map every failure onto aicore errors so that `LLMClient` can
decide what to retry.

## Options considered

1. **The official `ollama` Python package.**
   Pro: typed helpers, maintained by Ollama.
   Con: an extra dependency for one POST call; its own client and
   exception types sit between us and the status codes we must map.
2. **The `openai` SDK (Software Development Kit) against Ollama's
   OpenAI-compatible `/v1` endpoint.**
   Pro: one SDK could cover several vendors.
   Con: heavy dependency; the SDK retries by default, which would
   stack with `LLMClient`'s retries (one call could become 4 x 3
   requests); errors are wrapped in OpenAI-specific types.
3. **Raw `httpx` against the native `/api/chat` endpoint.**
   Pro: `httpx` is already a dependency; HTTP (Hypertext Transfer
   Protocol) status codes and network errors are visible directly;
   `httpx.MockTransport` gives offline tests without mocking our own
   code.
   Con: we own the request and response shapes and must follow
   Ollama API changes ourselves.

## Decision

Option 3. Supporting details:

- The response is parsed with a private Pydantic model. Unknown fields
  are ignored, and missing token counts default to 0.
- The `httpx` timeout is `None`. `LLMClient` owns timeouts in one
  place, and the 5 s `httpx` default would fail every cold model load.
- The constructor accepts an optional `transport`, so tests inject
  `httpx.MockTransport`.
- Only `Retry-After` in seconds is parsed. The HTTP-date form falls
  back to the computed backoff.

## Trade-offs

- About 150 lines we maintain, versus a library call.
- No streaming, which matches the rest of aicore today.
- The provider holds a connection pool, but the protocol has no
  `aclose` yet. Revisit when a long-running service needs clean
  shutdown.
