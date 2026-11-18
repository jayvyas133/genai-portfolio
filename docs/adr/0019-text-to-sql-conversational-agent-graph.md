# ADR (Architecture Decision Record) 0019: Text-to-SQL conversational agent graph

- **Status:** accepted
- **Date:** 2026-11-18
- **Scope:** 03-text-to-sql-agent

## Context

The M1 baseline (ADR 0018) showed both problems the agent must solve.
On the dev split it answered 0.82 of stand-alone questions but only 0.11
of follow-ups, because it has no memory: "Only for 2024." became a query
about artists. And 4 of its 5 execution errors named their own fix
(`column t.title does not exist`, `column reference "album_id" is
ambiguous`), but nothing fed the error back. A chat user also needs an
answer in words, not only a table.

Forces: a local 3B model on CPU (every extra call is seconds), a
bounded cost per turn, and an eval interface (`ask(thread_id,
question)`) that must not change.

## Options considered

### How a follow-up gets its context
1. **Rewrite the question first** (an extra call turns "Only for 2024."
   into a stand-alone question). Logs are easy to read, but every turn
   pays one more call, and the rewritten question still loses the exact
   SQL that the follow-up modifies.
2. **Replay earlier turns as chat messages** (chosen): each earlier
   question as a user message and its SQL plus a 5-row preview as the
   assistant reply. One call, and the model edits the previous query,
   which is what a follow-up asks for. This is the format chat models are
   trained on.
3. **Whole conversation in the prompt.** Simplest, but the prompt grows
   without bound. A window of the last 3 turns covers references to the
   previous answer or two.

### Self-correction
1. **No retry** (the baseline).
2. **A separate repair prompt.** A second prompt to maintain, and the
   model loses the conversation it was in.
3. **Append the failure to the same conversation** (chosen): the failed
   SQL as the assistant's reply, then "That query failed: <error>. Write
   a corrected query." Generate and repair become one node, and a second
   repair sees both earlier attempts. Guard rejections, Postgres errors
   and replies with no SQL are all repaired the same way.

### Memory store
1. **In-memory checkpointer now, Postgres checkpointer in M3** (chosen).
   Persistence only matters once requests from different processes
   share a conversation, which starts with the HTTP API.
2. **Postgres checkpointer now.** Adds a database, a role and a
   dependency before anything needs them.

## Decision

- `agent.py`: a LangGraph `StateGraph` run once per turn:
  `write_sql -> execute -> answer -> record`, with `execute` routing back
  to `write_sql` on any failure until `max_attempts` (default 3: one
  try, two repairs), and to `record` when out of attempts.
- `record` appends the turn (question, SQL, 5-row preview or the error)
  to `history`, trimmed to `history_turns` (default 3). Failed turns are
  remembered with their error, so a follow-up such as "try again with
  the album table" has context.
- State is plain data (str, int, lists and dicts; `QueryResult` is
  stored as `model_dump()`), so any checkpointer can serialise it,
  including the Postgres one in M3.
- Per-turn fields are reset by the input of each run; only `history`
  carries over through the checkpointer, keyed by `thread_id`.
- `answer`: one call that sees the question, the SQL and up to 20 rows
  and writes 1–3 sentences. An empty result gets a fixed reply, and a
  turn that ran out of attempts gets an apology written in code, with no
  call.
- Budget: at most `max_attempts + 1` calls per turn (4 by default).
- `AgentTurn` gains `answer` and `attempts`; the eval reports SQL
  attempts per turn and the share of turns that ran only after a
  repair, so the gain splits into memory (follow-up accuracy) and repair
  (fewer errors) without separate ablation runs.
- The system prompt describes follow-ups generically. An earlier draft
  gave concrete examples ("only for 2024"); on the smoke run the 3B model
  then added a year filter to a stand-alone question.

## Trade-offs

- **The answer call roughly doubles latency per turn** on a CPU model.
  It is what a chat user reads, so it stays; the eval still scores rows,
  not prose, and the answer is not graded yet.
- **The 5-row preview leaks a few result values into the next prompt.**
  That is the point (it lets "their invoices" find the customer), but on
  wide results it costs tokens; the window and row cap bound it.
- **Repair can turn a wrong-but-running query into nothing better.** Only
  failures are repaired; a query that runs and returns the wrong rows is
  not detected. Catching that would need a checker call per turn.
- **Memory is lost when the process exits** until M3 adds the Postgres
  checkpointer.
