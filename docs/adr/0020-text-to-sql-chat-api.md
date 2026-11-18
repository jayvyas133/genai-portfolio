# ADR (Architecture Decision Record) 0020: Text-to-SQL chat API, durable memory and the sync/async bridge

- **Status:** accepted
- **Date:** 2026-11-18
- **Scope:** 03-text-to-sql-agent

## Context

M2 (ADR 0019) built the agent with in-process memory. Serving it over
HTTP (Hypertext Transfer Protocol) changes the requirements:

- **Several processes.** gunicorn runs 2 workers; consecutive messages
  of one conversation can land on different workers, and a restart must
  not wipe conversations. Memory has to live outside the process.
- **Flask is synchronous; everything below it is async** (aicore,
  psycopg, LangGraph). Connection pools and the model client are bound
  to the event loop that created them.
- **Turns are slow and stateful.** A turn is 1–4 model calls (20–90 s
  on the local CPU model). Two turns of the same conversation running at
  once would both start from the same checkpoint, and one would
  overwrite the other's history.
- **Least privilege must survive the new database.** The memory store
  needs writes; the agent's SQL must still never run with write rights.

## Options considered

### Calling async code from Flask views
1. **`asyncio.run` per request** (DocuMind, ADR 0013). Simple, but a
   fresh loop per request means a new database connection and model
   client per request: pools cannot be shared.
2. **An async framework** (FastAPI, Quart). Removes the bridge, but the
   portfolio decision (ADR 0014) is Flask here, and many production Flask
   apps face exactly this problem.
3. **One event-loop thread per worker process** (chosen). The pools,
   model client and compiled graph are created once on that loop; views
   submit coroutines with `run_coroutine_threadsafe` and wait with a
   timeout. On timeout the coroutine is cancelled, so its work and locks
   are released.

### Where conversation memory lives
1. **Redis.** Already in the stack for 02, but LangGraph's official
   Redis saver is a separate module, and Postgres is already required
   here.
2. **Postgres, LangGraph's `AsyncPostgresSaver`** (chosen), in its own
   database `text_to_sql_memory` owned by its own role `sql_agent_app`.
   The agent's SQL still runs only as `sql_agent_ro` on `chinook`; the
   memory role cannot even connect to `chinook`.

### One turn at a time per conversation
1. **A per-process lock** (`asyncio.Lock` per id). Does not hold across
   gunicorn workers.
2. **A Postgres advisory lock** (chosen):
   `pg_try_advisory_lock(namespace, hashtext(id))` on a memory-pool
   connection held for the turn. It holds across every worker; a second
   request gets 409 immediately instead of queueing behind a 60 s turn.

## Decision

- `api.py`: `POST /conversations` (201, server-generated UUID),
  `POST /conversations/<id>/messages` (answer, SQL, columns, rows,
  attempts, error), `GET /conversations/<id>` (remembered turns),
  `GET /healthz` (both databases). Pydantic validates bodies (question
  1–2,000 characters, unknown fields rejected, 16 KB body cap).
- Errors: 404 unknown id, 409 turn already running, 502 model failure,
  503 database unreachable (including pool timeouts), 504 over
  `TEXT_TO_SQL_REQUEST_TIMEOUT_S` (120 s, below gunicorn's 150 s). A turn
  whose SQL never succeeded is still a 200: the conversation worked and
  the reply explains the failure.
- `service.py`: `ChatService` owns two pools, one per role. The memory
  pool is **twice** the data pool size: each running turn holds one
  memory connection for its lock while the checkpointer needs another,
  so equal sizes would deadlock at full load.
- A `conversation` table records issued ids, so an empty new
  conversation (200) is told apart from an id never issued (404). Ids
  are validated as UUIDs before any query.
- `init-memory` creates the role and database and runs the saver's own
  migrations. Like `load-db` it runs once per deploy, as a one-shot
  compose job (`text-to-sql-init`) that the API waits for
  (`service_completed_successfully`). The API container gets only the
  two agent URLs, never the admin URL.
- The executor takes an optional pool (the data pool in the API; a
  connection per query in the CLI and eval).

## Trade-offs

- **No streaming.** The reply arrives after the whole turn. Streaming
  the answer (Server-Sent Events, as in 02) is the next UX step.
- **A cold model can exceed the timeout.** The first request after
  Ollama loads the model took over 120 s and returned 504 (measured);
  warm turns took 40–54 s on CPU. A hosted model takes a few seconds.
- **409 instead of queueing.** Clients must retry; a chat UI disables
  "send" while a turn runs anyway.
- **Not built:** authentication and per-user conversation ownership,
  rate limiting, deleting conversations, trimming old checkpoints (the
  saver keeps every checkpoint of a thread). Each is needed before real
  users.
- **Answers are still ungraded.** In the smoke run one answer skipped a
  country because the rows were unsorted; the rows and SQL in the same
  response were correct.
