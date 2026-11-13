# ADR (Architecture Decision Record) 0003: Cloudflare Workers AI adapter

- **Status:** accepted
- **Date:** 2026-11-13
- **Scope:** aicore

## Context

We want a hosted LLM (Large Language Model) that is free for a
portfolio's traffic, so demos do not depend on a laptop running Ollama.
Cloudflare Workers AI gives 10,000 neurons per day on the free plan and
stops (rather than charging) when they run out. Per ADR 0001, the
adapter must map every failure onto aicore errors.

Workers AI does not classify failures by HTTP (Hypertext Transfer
Protocol) status alone. HTTP 429 with error code 3040 is a temporary
capacity limit, while HTTP 429 with code 3036 means the daily allowance
is used up until 00:00 UTC (Coordinated Universal Time).

## Options considered

1. **The OpenAI-compatible endpoint with the `openai` SDK (Software
   Development Kit).**
   Pro: one SDK shape for several vendors.
   Con: heavy dependency; the SDK retries by default, which stacks with
   `LLMClient`; Cloudflare's error codes are hidden behind OpenAI
   exception types.
2. **Raw `httpx` against the OpenAI-compatible endpoint.**
   Pro: familiar request shape.
   Con: Cloudflare-specific fields, such as `neurons` in usage, are less
   certain to be present.
3. **Raw `httpx` against the native REST (Representational State
   Transfer) endpoint `/accounts/{id}/ai/run/{model}`.**
   Pro: same approach as the Ollama adapter (ADR 0002); the standard
   Cloudflare envelope (`result`, `success`, `errors`) exposes error
   codes.
   Con: we own the request and response shapes.

## Decision

Option 3. Supporting details:

- **Error code beats status code** where they disagree on retryability:
  code 3036 raises a new `QuotaExhaustedError` (a `NonRetryableError`),
  so the client does not retry a quota that resets tomorrow. All other
  failures use the shared status mapping.
- **Shared HTTP helpers.** The status-to-error mapping and `Retry-After`
  parsing moved from the Ollama adapter to `providers/_http.py` when
  Cloudflare became the second user. Extracting error text stays per
  vendor, because each vendor formats error bodies differently.
- **`success: false`** on a 2xx response raises `InvalidRequestError`.
  It is not retried, because retrying blindly on a metered API can burn
  quota.
- **Reported model** is the one that served the request
  (`result.model`), falling back to the requested one. Cloudflare may
  route a model to a faster variant.
- **Configuration:** the token reuses `AICORE_API_KEY`; the account ID
  is a new, non-secret setting. The factory fails fast, naming every
  missing variable, and treats an empty value as missing.

## Trade-offs

- A second adapter of about 200 lines to maintain.
- Only code 3036 is special-cased. Others (for example 5035, a model
  that needs a paid plan, sent as 403) fall back to the status mapping,
  which is still non-retryable but less precisely named.
- No streaming, and no AI (Artificial Intelligence) Gateway caching yet.
- Neurons are returned in usage but not recorded until M2 (cost
  tracking).
