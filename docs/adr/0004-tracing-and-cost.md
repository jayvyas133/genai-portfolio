# ADR (Architecture Decision Record) 0004: Tracing and cost tracking

- **Status:** accepted
- **Date:** 2026-11-13
- **Scope:** aicore

## Context

Every project README must report latency and cost, and every LLM (Large
Language Model) call goes through `LLMClient`. We therefore need a record
of each call (how long it took, how many attempts, whether it failed,
tokens used, cost) that works the same way for every provider and can be
tested without a network. Langfuse is already in the Docker Compose
stack, but tying the client to one vendor's SDK (Software Development
Kit) would repeat the problem ADR 0001 solved for providers.

## Options considered

1. **Call the Langfuse SDK directly from `LLMClient`.**
   Pro: a trace UI (User Interface) immediately.
   Con: a hard dependency for every project and test; hard to unit-test;
   vendor lock-in at the core of the library.
2. **OpenTelemetry spans.**
   Pro: an industry standard with many exporters.
   Con: a large API (Application Programming Interface) surface and
   setup cost for one record per call; LLM-specific fields (tokens,
   neurons) still need custom attributes.
3. **Our own `Tracer` protocol that receives a Pydantic `CallRecord`.**
   Pro: same pattern as `LLMProvider`; trivial to test with an in-memory
   tracer; any backend (logs, Langfuse, OpenTelemetry) can be an adapter.
   Con: we own the record schema and the exporters.

## Decision

Option 3. Supporting details:

- **One record per logical call**, not per attempt. Retries are visible
  through `attempts`; latency includes backoff sleeps, because that is
  the wait the caller experienced.
- **Sent from a `finally` block**, so success, final failure, permanent
  error and cancellation (`asyncio.CancelledError`) each produce exactly
  one record. `outcome` is `"ok"` or the error class name.
- **Tracing never breaks a call.** Building or sending the record is
  wrapped in `try`; a failure is logged as a warning and the real result
  or error is returned unchanged.
- **`record` is synchronous.** It must be cheap (a list append, a log
  line); a slow backend should buffer and send in the background.
- **`NullTracer` is the default** (Null Object pattern), so the client
  never branches on "is tracing on".
- **Cost is computed in aicore** (`pricing.py`) from reported usage, not
  taken from the provider. Local providers cost `0.0`; Cloudflare is
  neurons × the list price (USD 0.011 per 1,000 neurons); anything
  without a known price is `None`, never `0`.
- **The provider reports its own name.** The first version took the name
  from `settings.provider`. A manual check showed the flaw: with
  `AICORE_PROVIDER=cloudflare` left in the shell, a call served by
  `FakeProvider` was recorded as Cloudflare with an unknown cost. The
  `LLMProvider` protocol now has a `name` attribute, so the record always
  names the object that actually made the call, and mypy flags any
  adapter that lacks it.

## Trade-offs

- No trace UI yet; the Langfuse exporter (M2.5) is parked for later.
- Validation failures in `complete_structured` are recorded as
  `outcome="ok"`, since the transport succeeded. Quality failures belong
  to the evaluation harness (M4).
- The price is a hard-coded list price and can go stale; the free plan
  is not charged, so the reported cost is "what this would cost on the
  paid plan".
- Ollama is reported as `0.0`: no per-call bill, although hardware and
  electricity are real costs.
- No per-token price table yet; it will be added with the first
  token-billed provider.
