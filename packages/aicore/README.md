# aicore

Shared platform layer used by all five portfolio projects. Every LLM
(Large Language Model) call in the portfolio goes through `aicore`, so
reliability, tracing, cost tracking and caching are built once and behave
the same everywhere.

## Status

| Milestone | Scope | State |
|---|---|---|
| M0 | Workspace, tooling, CI (Continuous Integration) skeleton | done |
| M1 | Provider-agnostic LLM client: retries, timeouts, structured output | done |
| M1.5 | Ollama adapter (local models) and provider factory | done |
| M1.6 | Cloudflare Workers AI adapter (hosted, free tier) | done |
| M2 | Observability and cost tracking | planned |
| M3 | Caching and guardrails | planned |
| M4 | Evaluation harness | planned |

## Problem

Each project needs to call an LLM. If every project calls a vendor SDK
(Software Development Kit) directly, then:

- retries, timeouts and error handling get re-implemented (and drift) per project;
- switching vendors means touching every call site;
- unit tests need network access or ad-hoc mocks.

## Architecture

```mermaid
flowchart LR
    S[LLMSettings] --> B[build_provider]
    B --> F[FakeProvider]
    B --> O[OllamaProvider]
    P[Project code] -->|LLMRequest| C[LLMClient]
    C -->|timeout + retry loop| I{{LLMProvider protocol}}
    I --> F
    I --> O
    B --> CF[CloudflareProvider]
    I --> CF
    CF -->|HTTPS| WA[(Cloudflare Workers AI)]
    O -->|HTTP| L[(Local Ollama server)]
    C -->|LLMResponse / Pydantic model| P
```

- **`LLMProvider`** (`providers/base.py`): a `typing.Protocol` with one
  method, `async complete(request) -> LLMResponse`. Adapters only translate
  to and from a vendor API (Application Programming Interface) and map vendor
  errors onto aicore's error types.
- **`LLMClient`** (`client.py`): the single entry point. It adds a timeout on
  every attempt, retries with exponential backoff and full jitter, and
  provides structured output.
- **`FakeProvider`** (`providers/fakeai.py`): a scripted, offline provider
  for tests and local development. It records every request in `.calls`.
- **`OllamaProvider`** (`providers/ollama.py`): calls a local Ollama
  server's `/api/chat` with `httpx` and maps HTTP (Hypertext Transfer
  Protocol) status codes and network failures onto aicore errors.
- **`CloudflareProvider`** (`providers/cloudflare.py`): calls Workers
  AI's native `/ai/run/{model}` endpoint with `httpx`. It reads
  Cloudflare's error codes as well as the status, so a used-up daily
  quota is not retried. The status mapping shared with Ollama lives in
  `providers/_http.py`.
- **`build_provider`** (`providers/factory.py`): reads
  `settings.provider` and returns the matching adapter. It is the only
  place that knows every adapter.
- **Errors** (`errors.py`): `RetryableError` (rate limit, timeout,
  unavailable) and `NonRetryableError` (for example, authentication or
  `QuotaExhaustedError`). Only retryable errors are retried.

## Usage

```python
import asyncio

from pydantic import BaseModel

from aicore.client import LLMClient
from aicore.models import LLMRequest, Message
from aicore.providers.fakeai import FakeProvider


class Book(BaseModel):
    """Target shape for the structured reply."""

    title: str
    year: int


async def main() -> None:
    """Run one plain and one structured call against the fake provider."""
    provider = FakeProvider(script=["Hello!", '{"title": "Dune", "year": 1965}'])
    client = LLMClient(provider)
    request = LLMRequest(messages=[Message(role="user", content="Hi")])

    reply = await client.complete(request)
    print(reply.text)  # Hello!

    book = await client.complete_structured(request, Book)
    print(book)  # title='Dune' year=1965


asyncio.run(main())
```

### Using a real provider

```python
from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.providers.factory import build_provider

settings = LLMSettings()  # AICORE_PROVIDER=ollama in .env
provider = build_provider(settings)
client = LLMClient(provider, settings)
```

Ollama setup: install it, run `ollama pull qwen2.5:0.5b`, then set
`AICORE_MODEL=qwen2.5:0.5b`. On a CPU-only (Central Processing Unit)
laptop, the first call loads the model into RAM (Random Access Memory)
and can take 10–30 s, so raise `AICORE_TIMEOUT_S` (for example, to 120).

Cloudflare setup: create a free account and an API token with
**Workers AI: Read** permission. Then set `AICORE_PROVIDER=cloudflare`,
`AICORE_MODEL=@cf/meta/llama-3.1-8b-instruct`, `AICORE_API_KEY=<token>`
and `AICORE_CLOUDFLARE_ACCOUNT_ID=<account id>`. The free plan allows
10,000 neurons per day and stops (no charge) when they run out; it
resets at 00:00 UTC (Coordinated Universal Time).

## Configuration

Settings are read from environment variables (or a `.env` file) with the
prefix `AICORE_`. They are validated at startup.

| Variable | Default | Meaning |
|---|---|---|
| `AICORE_PROVIDER` | `fake` | Which adapter to use: `fake`, `ollama` or `cloudflare` |
| `AICORE_MODEL` | `fake-model` | Model used when a request does not set one |
| `AICORE_API_KEY` | unset | Provider key or token (Cloudflare API token), stored as a `SecretStr` so it is never printed |
| `AICORE_TIMEOUT_S` | `30.0` | Timeout per attempt, in seconds |
| `AICORE_MAX_RETRIES` | `3` | Retries after the first attempt (so 4 calls at most) |
| `AICORE_OLLAMA_BASE_URL` | `http://localhost:11434` | Root URL of the Ollama server |
| `AICORE_CLOUDFLARE_ACCOUNT_ID` | unset | Cloudflare account ID; required when the provider is `cloudflare` |

## Reliability behaviour

- **Retries:** only `RetryableError` is retried. A permanent error, such as a
  bad API key, is raised immediately because resending the request cannot
  fix it.
- **Backoff:** full jitter. The delay is a random value between 0 and
  `min(max_delay, base_delay * 2^attempt)`. The randomness keeps many clients
  from retrying in lockstep.
- **`Retry-After`:** if a provider says how long to wait, that value is used
  instead of the computed backoff.
- **Timeouts:** applied to each attempt, not to the whole call. A hung
  attempt becomes `ProviderTimeoutError` and is retried.
- **Structured output:** the JSON (JavaScript Object Notation) Schema of the
  target Pydantic model goes in as a system message. The reply is stripped
  of Markdown code fences and then validated. A reply with the wrong shape
  raises `StructuredOutputError`, which keeps the raw text. It is **not**
  retried.

## Ollama error mapping

| Failure | aicore error | Retried? |
|---|---|---|
| Connection refused / network error | `ProviderUnavailableError` | yes |
| HTTP client timeout | `ProviderTimeoutError` | yes |
| 429 | `RateLimitError` (reads `Retry-After` seconds) | yes |
| 5xx | `ProviderUnavailableError` | yes |
| 401 / 403 | `AuthenticationError` | no |
| 404 (unknown model) and other 4xx | `InvalidRequestError` | no |

## Cloudflare error mapping

| Failure | aicore error | Retried? |
|---|---|---|
| Network error / HTTP client timeout | `ProviderUnavailableError` / `ProviderTimeoutError` | yes |
| 429, code 3040 (temporary capacity limit) | `RateLimitError` | yes |
| 429, code 3036 (daily allowance used up) | `QuotaExhaustedError` | no |
| 5xx | `ProviderUnavailableError` | yes |
| 401 / 403 | `AuthenticationError` | no |
| Other 4xx, or `success: false` on a 2xx | `InvalidRequestError` | no |

## How to run the tests

From the workspace root:

```powershell
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Unit tests use `FakeProvider` or `httpx.MockTransport`, so they need no
network access and no API key. Integration tests are skipped by default.
They call a local Ollama and, if credentials are set, the real Cloudflare
API (a few neurons per run):

```powershell
uv run pytest -m integration -v
```

## Trade-offs and future work

- Two real adapters: Ollama (local) and Cloudflare Workers AI (hosted).
  Gemini is parked.
- Only Cloudflare code 3036 is special-cased; other codes use the shared
  status mapping.
- The `LLMProvider` protocol has no `aclose`, so code holding a plain
  `LLMProvider` cannot close an HTTP adapter's connection pool. This
  will be fixed when a long-running service needs clean shutdown.
- `Retry-After` in HTTP-date form is ignored; the computed backoff is
  used instead.
- Validation failures are not retried. A "repair" retry, which sends the
  validation error back to the model, may be added once eval data shows it
  is worth the extra cost.
- Latency and cost numbers will be added in M2, when tracing lands.
- See [ADR 0001](../../docs/adr/0001-provider-interface.md) for why the
  client is built around a provider protocol, and
  [ADR 0002](../../docs/adr/0002-ollama-adapter-raw-httpx.md) for why the
  Ollama adapter uses raw `httpx`, and
  [ADR 0003](../../docs/adr/0003-cloudflare-workers-ai-adapter.md) for the
  Cloudflare adapter and its error-code mapping.