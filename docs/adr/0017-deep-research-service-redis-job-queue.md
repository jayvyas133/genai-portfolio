# ADR (Architecture Decision Record) 0017: Deep Research service with a Redis job queue

- **Status:** accepted
- **Date:** 2026-11-18
- **Scope:** 02-deep-research

## Context

M3 turns the agent graph (ADR 0016) into an HTTP (Hypertext Transfer
Protocol) service. The forces were:

- **One report takes minutes** (249 s median on the development machine,
  up to 7 LLM (Large Language Model) calls). An HTTP request held open
  that long breaks behind load balancers and proxies, ties up a server
  worker, and loses the work if the connection drops.
- **Users want to see progress** during those minutes, not a spinner.
- **Work must survive restarts.** Deploys and crashes happen during long
  jobs; a queued or half-done job should not silently vanish.
- **The model backend is the bottleneck.** A local Ollama serves one
  request at a time, so the service needs back-pressure, not unlimited
  concurrency.

Products that do this (OpenAI deep research in background mode,
LangGraph Platform runs) all use the same shape: submit a job, get an
id, follow or poll it. The decisions were where jobs run, how progress
reaches clients, and how the queue behaves under failure.

## Options considered

### Where jobs run
1. **In the API process (`asyncio` tasks + in-memory registry).**
   Fewest moving parts. But jobs die with every deploy or crash, the API
   cannot run more than one replica (state is in one process), and the
   API's memory and CPU (Central Processing Unit) are shared with the
   research.
2. **Celery with a Redis broker.** The best-known Python task queue. But
   it is sync-first while the graph and aicore are `async`; its progress
   reporting is a state string, not an event stream; and it adds a large
   dependency for one task type.
3. **A small async queue library (Arq, SAQ).** Async and Redis-based.
   Still needs a separate progress channel for streaming, and its retry
   and abort semantics would be one more thing to learn and explain.
4. **Redis Streams with a consumer group, own worker (~150 lines).**
   Streams give what a job queue needs natively: `XREADGROUP` hands each
   entry to exactly one consumer, `XACK` marks it done, unacknowledged
   entries stay in the pending list, and `XAUTOCLAIM` lets another
   consumer take over entries a dead worker left behind. The same data
   type stores per-job progress events, whose ordered ids map directly to
   SSE (Server-Sent Events) `Last-Event-ID`. One dependency (`redis`,
   already used by aicore's cache) and one service (Redis, already in
   compose).

### Progress to clients
1. **Polling `GET /research/{id}`.** Simple, but slow to show progress
   or wasteful when polled fast.
2. **WebSockets.** Two-way, but nothing flows from client to server here;
   needs its own reconnect logic and is harder to put behind proxies.
3. **SSE.** One-way server-to-client over plain HTTP, browsers reconnect
   automatically and send `Last-Event-ID`; the format LLM streaming APIs
   use.

## Decision

**Redis Streams queue + separate worker process, SSE for progress,
FastAPI for the API.**

- `POST /research` validates the request, writes the job record and
  `XADD`s it to `research:queue`, and returns `202 Accepted` with a
  `Location` header. The API holds no state, so it can run as many
  replicas as needed.
- The **worker** (`deep-research worker`) reads with `XREADGROUP` in the
  `workers` group. For each job it marks it running, runs the graph under
  a hard timeout (`DEEP_RESEARCH_JOB_TIMEOUT_S`, 900 s), publishes each
  finished graph step to `research:job:<id>:events`, writes the final
  state, then `XACK`s and `XDEL`s the entry.
- **Delivery is at-least-once.** If a worker dies mid-job, the entry stays
  pending; any worker claims it with `XAUTOCLAIM` once it has been idle
  for the job timeout plus 60 s. Because no job can legitimately run past
  the timeout, a live job is never claimed twice. `attempts` caps it at 2
  starts, so a job that kills workers (a "poison pill") fails instead of
  looping.
- **Graceful shutdown** (SIGTERM from Docker) cancels the running graph
  and hands the job straight back to the queue without counting the
  attempt, instead of waiting ~16 minutes for the claim timeout.
- **State changes are check-and-set under `WATCH`** (optimistic
  locking). Example race: the API cancels while the worker finishes.
  Without it, the API could write "running, cancel requested" over
  "done" and the job would hang as running forever.
- **Cancel:** a queued job is cancelled at once (the worker skips its
  entry later); a running job gets `cancel_requested`, which the worker
  polls every second and then cancels the graph task.
- **Back-pressure:** queue length (queued + running, since done entries
  are deleted) above `DEEP_RESEARCH_MAX_PENDING_JOBS` gives `429 Too Many
  Requests` with `Retry-After`. `DEEP_RESEARCH_WORKER_CONCURRENCY`
  (default 1) sets jobs per worker process.
- **SSE** replays the job's events from the start (or after
  `Last-Event-ID`), then blocks on `XREAD` for new ones, sends a
  keep-alive comment every 15 s so proxies keep the connection open, and
  closes after the final event (`done`, `failed`, `cancelled`).
- **Redis durability:** AOF (append-only file) on, fsync every second.
  Finished jobs and their events expire after `DEEP_RESEARCH_RESULT_TTL_S`
  (1 day).
- **Docker:** one image, two compose services (`deep-research-api` on
  port 8001, `deep-research-worker` with `restart: unless-stopped`).

### Found in the real runs

Unit tests use fakeredis, which never times out. Against real Redis,
redis-py 8's default 5 s socket read timeout made idle blocking reads
(15 s SSE wait, 5 s worker wait) raise `TimeoutError`; that killed the
first worker and cut SSE streams. Fix: `jobs.connect()` sets a 30 s read
timeout (above the longest 15 s block), and a timeout during an idle
blocking read is treated as "nothing new" rather than an error, because a
CPU-starved host (Ollama on the same machine) can stretch a wait past
any fixed limit. A regression test pins the timeout to the block
lengths.

## Trade-offs

- **We own ~150 lines of queue code** instead of a library's tested
  edge cases. Mitigated by tests for every failure path (crash reclaim,
  shutdown requeue, cancel races, timeout, poison pill) and by staying on
  Redis's own primitives. If more task types appear, Arq or Celery becomes
  worth it.
- **At-least-once, not exactly-once:** a reclaimed job reruns from the
  start and its event stream shows a second `started`. Research has no
  side effects outside Redis, so a rerun is safe. Checkpointing the graph
  (LangGraph checkpointer) would resume instead, at the cost of state
  storage per step.
- **Crash recovery is slow by design:** a crashed worker's job waits for
  the timeout plus 60 s (16 minutes by default). A heartbeat key per
  running job would detect death in seconds; not worth it at this scale.
- **The queue limit is soft:** check-then-add can overshoot by a few
  under concurrent submits. A Lua script would make it exact.
- **Each open SSE stream holds one Redis connection** while it blocks.
  Fine for hundreds of viewers; past that, one reader per API process
  fanning out to clients (or Redis pub/sub) would be the fix.
- **Cancel latency is up to 1 s** (polling). Pub/sub would make it
  instant but adds a subscriber per running job.
- **Error messages are returned as-is** (`RuntimeError: ...`), useful
  here but they can leak internals; a public API would map them to codes.
- **No authentication or per-user limits:** needed before exposing the
  API beyond a private network.
