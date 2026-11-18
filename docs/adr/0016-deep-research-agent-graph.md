# ADR (Architecture Decision Record) 0016: Deep Research agent graph

- **Status:** accepted
- **Date:** 2026-11-18
- **Scope:** 02-deep-research

## Context

M1 measured a single-shot baseline: one search with the raw question,
one LLM (Large Language Model) call. On a 3-question smoke run it found
the right articles (7/7) but the passages holding only 6 of 17 key facts,
and the small writer model cited nothing (0/14 sentences). M2 builds the
multi-agent system that should close those gaps. The forces were:

- **Research questions have several parts** ("compare X and Y on launch,
  orbit and mirror"); one query cannot target all of them.
- **Cost and latency must stay bounded.** Every agent step is an LLM
  call; on the development machine (CPU-only Ollama, `qwen2.5:3b`) one
  call takes 20–80 s.
- **Small models break structured output now and then.** One bad JSON
  reply must not fail a whole run.
- **aicore has no tool-calling support**, and adding it is a large
  change to every provider adapter.

## Options considered

### Control flow
1. **One agent in a tool-calling loop** (ReAct, Reasoning and Acting):
   the model decides when to search and when to stop. Flexible, but the
   number of calls is unbounded unless capped, small models loop or stop
   early, and it needs tool calling in aicore.
2. **Fixed graph of specialised steps** (chosen): plan, parallel
   research, write, critique, optional rewrite. The LLM decides *what*
   to search; the code decides *that* it searches. Every run has a known
   maximum number of calls.
3. **Fully autonomous multi-agent chat** (agents message each other
   freely). Hard to bound, hard to test, hard to evaluate.

### Framework
1. **Plain `asyncio`** (`gather` for the fan-out, a loop for revision).
   Fewer dependencies, but state merging, conditional routing and
   per-step streaming are hand-written.
2. **LangGraph** (chosen). The graph is explicit (nodes, edges,
   conditional edges), `Send` fans out one branch per sub-question and
   runs them concurrently, a reducer (`operator.add`) merges their
   results, and `astream(stream_mode="updates")` gives per-step progress
   for free, which M4's service will stream to clients.

### Critic
1. **LLM review only.** Catches coverage gaps, but a small model's
   review is unreliable.
2. **Code checks only.** Deterministic, but cannot judge coverage.
3. **Both** (chosen). Code checks what code can check (every sentence
   cited, no marker pointing at a missing source); the LLM checks
   coverage of the planned sub-questions. Either can ask for a rewrite.

## Decision

```
START -> plan -> research x N (Send, parallel) -> write
      -> critique -> (fixes needed and budget left: write | else: END)
```

- **Planner**: one structured call returns at most `max_sub_questions`
  (3) self-contained sub-questions.
- **Researcher** (one branch per sub-question): one structured call
  writes up to `queries_per_sub_question` (2) keyword queries; code runs
  each through `SearchProvider` and merges results **round-robin** (the
  1st hit of each query, then the 2nd...), keeping
  `passages_per_sub_question` (4). BM25 (Best Matching 25) scores from
  different queries are not on one scale, so sorting by score would let
  one query crowd out the others.
- **Writer**: the same writer and prompt as the baseline (`writer.py`),
  plus the sub-questions as an outline and, on a rewrite, the previous
  draft and the critic's fixes. Sharing the writer means a difference in
  eval numbers comes from planning and retrieval, not from wording.
- **Critic**: code checks plus one structured LLM review. The rewrite
  is not reviewed again once the revision budget (`max_revisions`, 1) is
  spent, so no call is wasted on a verdict that cannot change anything.
- **Budgets**: at most `2 + max_sub_questions + 2 * max_revisions` LLM
  calls per report (7 with the defaults). All limits are settings
  (`DEEP_RESEARCH_*`).
- **Fallbacks**: invalid planner JSON → the question itself is the only
  sub-question; invalid query JSON → the sub-question is the query;
  invalid critic JSON → code checks only. Quality degrades, the run
  completes.
- The graph returns the same `Report` as the baseline, so the M1 eval
  scores both, and the CLI (`--researcher single-shot|agents`) picks one.

## Trade-offs

- **The critic can only ask for a rewrite, not for more research.** If
  search missed a fact, rewriting cannot add it. Sending the critic's
  gaps back to the researchers is the natural next step, at the cost of
  more calls; add it if evals show "facts retrieved" is the bottleneck
  after planning improves.
- **A fixed graph cannot adapt its depth** to an easy or hard question;
  a simple question still gets planned and reviewed. Budgets keep that
  waste small.
- **Parallel branches are only parallel with a parallel backend.**
  Ollama on CPU serves one request at a time, so locally the branches
  queue; on Cloudflare they overlap.
- **More calls, more latency**: about 3–4x the baseline's latency
  locally (7 calls against 1). The eval reports both so the trade can be
  judged.
