# ADR (Architecture Decision Record) 0005: Response caching

- **Status:** accepted
- **Date:** 2026-11-14
- **Scope:** aicore

## Context

The portfolio projects send the same prompt many times: evaluation runs
replay a golden dataset, demos repeat the same questions, and tests run
again and again during development. Every repeat of an LLM (Large
Language Model) call costs latency (about 0.8 s on Cloudflare Workers AI)
and, on a paid plan, money. On the Cloudflare free plan it also uses up
the daily neuron allowance. A cache must be:

- **correct**: it must never return a reply to a different request;
- **shared**: several processes (a web app, an eval run) should reuse
  one another's replies, and the cache should survive restarts;
- **safe**: a cache outage must not break LLM calls;
- **testable** without a network, like the provider and tracer layers.

## Options considered

1. **In-process dictionary or `functools.lru_cache`.**
   Pro: no infrastructure, fastest possible lookup.
   Con: not shared between processes, lost on restart, no expiry.
   `lru_cache` also cannot wrap an `async` method usefully.
2. **Redis, called directly from `LLMClient`.**
   Pro: shared, survives restarts, built-in expiry (TTL, time to live).
   Con: every unit test would need Redis or mocks of the Redis client.
3. **A `ResponseCache` protocol with two backends: `InMemoryCache`
   (tests, local runs) and `RedisCache` (shared).**
   Pro: same pattern as `LLMProvider` (ADR 0001) and `Tracer` (ADR 0004);
   unit tests use the in-memory backend; another backend (for example
   Postgres) is one small class.
   Con: one more abstraction to own.
4. **Semantic cache (match similar prompts by embedding distance).**
   Pro: also hits on paraphrased prompts.
   Con: needs an embedding call on every request and a similarity
   threshold. A wrong match returns a confidently wrong answer, and
   tuning the threshold needs eval data we do not have yet.

## Decision

Option 3, with Redis as the shared backend. Supporting details:

- **Exact-match key.** `cache_key` hashes (SHA-256, Secure Hash
  Algorithm) the provider name plus the *whole* request, serialized as
  JSON (JavaScript Object Notation) with sorted keys. Any field that can
  change the reply (messages, model, temperature, `max_tokens`) changes
  the key. The client fills in the default model *before* building the
  key, so changing `AICORE_MODEL` can never reuse another model's
  replies. Keys start with `aicore:v1:`; bumping the version invalidates
  every old entry at once.
- **Only `temperature == 0` is cached.** A sampled reply is meant to
  vary; caching it would silently remove that variation.
- **Cache errors are a miss.** Read and write failures (Redis down,
  corrupt JSON) are logged as warnings and the provider is called as if
  there were no cache. The cache can only make a call faster, never make
  it fail. An integration test writes a corrupt entry and checks that it
  is replaced by the fresh reply.
- **Only successful replies are stored**; errors are never cached.
- **Traced like any other call.** A hit still sends one `CallRecord`,
  with `cached=True`, `attempts=0` and `cost_usd=0.0`, so cost reports
  show what the cache saved.
- **TTL in the backend.** `RedisCache` stores each entry with
  `SET key value EX ttl` in one command. The default TTL is one day
  (`AICORE_CACHE_TTL_S=86400`), so a model update on the provider's side
  reaches us within a day.
- **Values are `LLMResponse` JSON** written with Pydantic, not pickle,
  so a value read from a shared store cannot run code and can be read
  from any language.

## Measured effect

`scripts/measure_llm.py` with Redis in Docker, Cloudflare
`@cf/meta/llama-3.1-8b-instruct`, 6 calls, on 2026-11-14:

| | Latency | Cost per call |
|---|---|---|
| Miss (provider call) | 798 ms | USD 0.0000175 |
| Hit (median of 5) | 1.3 ms | USD 0 |

A hit is roughly 600× faster. The first call after startup also pays for
opening the Redis connection (about 100 ms with the fake provider), so a
long-running service should open it at startup.

## Trade-offs

- Exact match only: a prompt that differs by one character is a miss.
  A semantic cache can be reconsidered once the evaluation harness (M4)
  can measure wrong-hit rates.
- Stale replies for up to one TTL after a provider changes a model
  behind the same name.
- No stampede protection: two processes that miss at the same moment
  both call the provider. That costs one extra call, not a wrong answer.
- Cached replies are stored in plain text in Redis. Prompts that contain
  personal or secret data need a short TTL, a private Redis, or no cache.
- `InMemoryCache` never expires and never evicts; it is for tests and
  short local runs only.
