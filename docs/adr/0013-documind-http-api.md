# ADR (Architecture Decision Record) 0013: DocuMind HTTP API

- **Status:** accepted
- **Date:** 2026-11-18
- **Scope:** 01-documind

## Context
DocuMind's pipeline (guard → retrieve → generate, ADR 0012) was reachable
only from the CLI (Command Line Interface). A portfolio demo needs an HTTP
(HyperText Transfer Protocol) API (Application Programming Interface) that
a reviewer can start with one command. The goal is a simple setup, not a
production platform.

Constraints:
- The pipeline is async (psycopg async connection, httpx client in aicore).
  Flask views are synchronous, and async objects are bound to the event
  loop that created them.
- psycopg's async mode cannot run on Windows' default Proactor loop.
- One Cloudflare call took 33 s in M5, so a request must have a ceiling.

## Options considered
1. **Web framework**
   - **FastAPI**: async-native, so a shared connection pool would be
     natural. But it's a second framework in the repo, and the project
     already declares Flask.
   - **Flask with `flask[async]` views**: adds `asgiref`, which still runs
     each async view on a new event loop, and on Windows needs a global
     event-loop policy change for psycopg.
   - **Flask with sync views calling `asyncio.run`** (chosen): the same
     helper the CLI already used (`aio.run`), with a selector loop on
     Windows. No extra dependency.
2. **Connections**
   - **A pool on one long-lived background loop** (`run_coroutine_threadsafe`):
     no per-request connect, but adds a thread, lifecycle and shutdown code.
   - **A connection and a model client per request** (chosen): each request
     owns everything it uses, so nothing is shared across loops or threads.
3. **Serving**
   - **Flask development server only**: simplest, but single-process and
     flagged by Flask itself as not for production.
   - **Dockerfile + gunicorn** (chosen, alongside `documind serve` for local
     development): one `docker compose up` needs no Python on the host.

## Decision
- `POST /ask` and `GET /health` in a Flask app factory,
  `create_app(open_session=None, timeout_s=None)`. `open_session` is the seam:
  production opens Postgres and an aicore client, and tests pass a fake
  retriever and the fake provider.
- Each request runs `answer_within` through `aio.run`:
  `asyncio.timeout(DOCUMIND_REQUEST_TIMEOUT_S)` (30 s) around opening the
  session and answering. Errors map to stable JSON codes: 400 invalid body,
  413 over 64 KB, 502 provider failed, 503 Postgres down, 504 timeout. A
  refusal is a 200. Provider messages are logged and never returned.
- The Docker image is built from the repository root (aicore is a workspace
  dependency) with `uv sync --frozen --package documind`. Dependencies go in
  a cached layer before the code, and the container runs as a non-root user.
  gunicorn runs 2 workers × 4 threads (gthread) with `--timeout 60`, above the
  app's 30 s, so the app answers 504 before gunicorn kills a worker.

Root cause: the async pipeline plus Flask means an event loop per request.
Given that, opening resources per request is the only arrangement with no
shared async state, and it costs 40–85 ms against a ~2 s answer.

## Trade-offs
- **Per-request overhead.** 40–85 ms per warm request (measured in the
  container with blocked questions) and 0.3–0.5 s for a worker's first
  request. No reuse of TLS (Transport Layer Security) connections to
  Cloudflare. Revisit with a pool on a long-lived loop, or FastAPI, if traffic
  or a latency budget makes it matter.
- **Concurrency** is 8 requests at once (2 × 4 threads), each holding one
  Postgres connection. That's ample for a demo. Postgres allows 100 connections
  by default.
- **Timeout is cooperative.** `asyncio.timeout` cancels awaiting I/O. A
  CPU-bound step would not be interrupted, but the pipeline has none.
- **Not covered:** authentication, rate limiting, streaming, and a fallback
  when rerank fails. A provider 429 surfaces as 502. All are listed as future
  work in the README.
- **End-to-end latency through the API was not measured.** The Cloudflare
  free-tier daily quota ran out during this milestone. The API's own
  overhead was measured, and answer latency is the CLI's (ADR 0012) plus that
  overhead.
