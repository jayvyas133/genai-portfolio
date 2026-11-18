# ADR (Architecture Decision Record) 0018: Text-to-SQL database, read-only execution and execution-accuracy evals

- **Status:** accepted
- **Date:** 2026-11-18
- **Scope:** 03-text-to-sql-agent

## Context

The Text-to-SQL (Structured Query Language) agent will be a chatbot: users
ask questions in plain English, follow up on earlier answers, and the
agent writes and runs SQL. Evals come first, so M1 builds what every later
milestone is measured with: a database, a safe way to run model-written
SQL, a golden dataset of conversations, an eval and a baseline. The
forces were:

- **Model-written SQL is untrusted input.** A model can be wrong, or be
  talked into `DELETE`. Whatever it writes must not change data, lock
  rows or run forever.
- **Many SQL queries give one right answer.** Grading query text would
  fail correct queries that are written differently.
- **Follow-ups are the point of a chatbot.** "Only for 2024." means
  nothing without the turn before it, and the eval has to show that.
- **No paid quota during development.** Runs use a local Ollama model
  (`qwen2.5:3b`, CPU only).

## Options considered

### Database
1. **SQLite file.** Zero setup, but production Text-to-SQL runs against a
   server database with roles and grants, which is where the safety
   story lives.
2. **Northwind.** Well known, but no maintained PostgreSQL script.
3. **Chinook on the existing Postgres** (chosen). A music store (11
   tables: customers, invoices, tracks, albums, artists, genres,
   employees, ...), MIT licence, official snake_case PostgreSQL script,
   invoices across 2021–2025 and 24 countries, so follow-ups like "only
   2024" or "by country" make sense.

### Keeping queries read-only
1. **Keyword filter** (reject text containing `DELETE`, `DROP`, ...).
   Misses `WITH d AS (DELETE ...) SELECT ...`, and rejects a column
   named `update`.
2. **Parser only** (sqlglot). Accurate, but if it has a gap nothing
   else stops the write.
3. **Database permissions only.** The real boundary, but Postgres's
   errors ("cannot execute DELETE in a read-only transaction") are a
   poor hint for a model repairing its query, and a misconfigured role
   would leave nothing behind it.
4. **Three layers** (chosen), each enforced by a different component:
   1. *Database role* `sql_agent_ro`: SELECT-only grants, no CREATE or
      TEMP, `default_transaction_read_only = on` and a role-level
      `statement_timeout`. This is the security boundary.
   2. *Guard* (`guard.py`, sqlglot): exactly one statement, the top
      level must be a query, and no write, DDL, `SET`, `SELECT INTO` or
      `FOR UPDATE` anywhere inside it. It gives the model a clear error.
   3. *Executor*: each query runs in a `READ ONLY` transaction with
      `SET LOCAL statement_timeout`, is always rolled back, and returns
      at most `max_rows` rows (flagged as truncated).

   Integration tests prove each layer holds *on its own*, with the
   layers in front of it bypassed.

### Scoring
1. **Exact SQL match.** Fails correct but differently written queries.
2. **LLM judge.** Costly, noisy, and needs its own validation.
3. **Execution accuracy** (chosen), the Spider and BIRD benchmark
   metric: run predicted and gold SQL and compare the results, in code.
   Relaxed for a chatbot: column order and names are ignored, extra
   columns are allowed (a genre *and* its count is a good answer),
   numbers are compared to 2 decimals, and row order only matters when
   the gold query has a top-level `ORDER BY`.

### Multi-turn scoring
1. **Gold history** (feed each turn the gold previous answers). Isolates
   one turn, but no user ever has a perfect history.
2. **Own history** (chosen, as in the CoSQL and SParC benchmarks). Each
   conversation gets a fresh thread id and the agent sees only its own
   earlier turns, so an early mistake can derail a follow-up, just as
   for a real user. A conversation passes only if every turn does.

## Decision

- `text-to-sql load-db` creates the database and role, idempotently,
  from the vendored script (`infra/postgres/init.sql` only runs on a new
  volume). The role's password comes from `TEXT_TO_SQL_DATABASE_URL`, and
  the loader refuses to use the admin user as the agent's role.
- The schema in the prompt is introspected live from `pg_catalog` as
  the read-only role, so it never drifts from the real tables and shows
  only what the agent may query. `information_schema` was not used: it
  hides constraints (and so foreign keys) from roles that do not own the
  tables.
- Golden dataset: 28 conversations, 42 turns (20 single questions and
  8 conversations of 2–3 turns; 17 dev / 11 test), tagged `join`,
  `aggregate`, `order-limit`, `follow-up`, `ellipsis`, `reference`, ...
  Gold queries avoid ties at a `LIMIT` and inside a compared `ORDER BY`.
  `check-golden` (and every eval run) executes all gold queries first
  and fails on errors, empty results or truncation.
- Baseline `single-shot`: schema DDL (Data Definition Language) in the
  system prompt, one call, SQL read from a fenced code block (not JSON:
  SQL's quotes and newlines are easy to mis-escape), then executed. No
  memory and no retries.
- The `Agent` protocol is `ask(thread_id, question) -> AgentTurn`, the
  same call a chat client makes. Memory is the agent's business, so M2
  can add a LangGraph checkpointer keyed by `thread_id` without the
  eval changing.

## Trade-offs

- **A sample schema.** 11 tables fit in the prompt whole; a real
  warehouse with hundreds of tables needs schema selection (M2 or later).
- **Relaxed matching can pass a wrong answer** in rare cases (for
  example `SELECT *` that happens to contain the gold columns). Extra
  columns are allowed on purpose; the per-turn output keeps the SQL, so
  a reviewer can check suspicious passes.
- **Gold queries pin one reading of each question.** "Top-spending
  customer" means by invoice total; another reasonable reading fails.
  Questions say which columns and order they want to keep this small.
- **No function denylist.** `pg_sleep` is stopped by the timeout, and
  `pg_read_file`-style functions need privileges the role does not have.
  `set_config` can switch the read-only default off for the session, but
  the transaction is already read-only and the grants still refuse every
  write (tested).
- **A connection per query** for now, which is fine for the CLI and the
  eval; the M3 HTTP API (Application Programming Interface) will use a
  pool.
- **Small eval runs on a 3B model.** Numbers show the shape (follow-ups
  fail without memory), not final quality; they will be rerun with a
  stronger model.
