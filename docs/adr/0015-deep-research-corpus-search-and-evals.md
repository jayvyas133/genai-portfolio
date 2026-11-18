# ADR (Architecture Decision Record) 0015: Deep Research corpus, search tool and report evals

- **Status:** accepted
- **Date:** 2026-11-18
- **Scope:** 02-deep-research

## Context

Deep Research will be a multi-agent system: a planner, parallel
researchers, a writer and a critic (M2–M3). Evals come first, so M1
builds what every later milestone is measured with: something to search,
a golden dataset, an eval runner and a baseline to beat. The forces were:

- **Reproducibility.** Two eval runs are only comparable if the system
  saw the same sources both times.
- **No paid quota during development.** The Cloudflare Workers AI free
  tier ran out on 2026-11-18; development runs on a local Ollama model
  (`qwen2.5:3b`, CPU only, about 80 s per report).
- **Reports are long-form.** A report can be right on four facts and
  silent on two; a single pass/fail or 1–10 score hides which.
- **Agent systems cost several times more calls than a single call**, so
  the eval must show whether the extra calls buy quality.

## Options considered

### What to search
1. **Live web search API** (for example Tavily). What production
   research agents use, but results change over time (evals stop being
   comparable), it needs an API key and the free tier is 1,000 searches a
   month.
2. **Live web plus record and replay** for evals. Production-faithful
   and reproducible, but more moving parts than M1 needs.
3. **A frozen snapshot of 38 Wikipedia articles, searched with BM25**
   (chosen for now). Offline, free and deterministic.

### How to rank
1. **Embeddings** (dense vectors). Better at paraphrase, but needs an
   embedding model call per passage and a vector store.
2. **BM25** (Best Matching 25, keyword ranking; chosen). About 50 lines
   of standard-library Python, indexes 2,483 passages in 0.2 s, about 5 ms
   per query. Agents write and rephrase their own queries, which
   compensates for its weakness with paraphrase.

### How to score a report
1. **One holistic judge score (1–10).** Cheap, but noisy and
   unexplainable.
2. **Key facts per question** (chosen). Each golden case lists 3–6
   atomic facts with the document and verbatim quote that states each.
   One judge call per report marks every fact stated or not, with a
   reason.
3. **Plus checks in code wherever possible:** whether each fact's quote
   was in a retrieved passage, citation markers that point nowhere,
   sentences with no citation.

## Decision

- `SearchProvider` protocol (`search(query, k) -> list[SearchHit]`)
  with `CorpusSearch` (BM25 over the snapshot) and `FakeSearch` (tests).
  Agents only see the protocol, so a live web adapter can be added later
  without changing them.
- Golden dataset: 13 research questions (8 dev, 5 test), each needing
  1–3 articles and 3–6 key facts (61 facts in total). A unit test checks
  that every quote really is in the corpus, so a missed fact is always
  the system's fault, not the dataset's.
- Metrics: key facts stated (judge), key facts retrieved (code),
  expected documents retrieved, sentences with a citation, invalid
  markers, and cited sentences supported by their passages (judge, up to
  `--max-claims` per report). Latency, LLM calls and cost per report are
  recorded, and judge cost is kept separate.
- "Facts stated" vs "facts retrieved" separates a search failure from a
  writing failure, which tells M2 which agent to improve.
- Every saved run records the model, the judge model and both prompt
  versions. **The judge model must stay fixed across runs that are
  compared.**
- Baseline: `single-shot` searches once with the raw question and makes
  one LLM call. It is the bar the agent graph must clear.
- The eval loop is project-specific, not `aicore.evals.run_eval`:
  aicore scorers are synchronous and model-free by design, and both
  judges here are model calls.
- The report is plain text with `[n]` markers, not JSON: escaping a
  long report inside a JSON string is a common failure for small models.

## Trade-offs

- **Not the live web.** The system answers from 38 articles. That is a
  narrower world than a real research agent's, and BM25 misses
  paraphrases that a web search engine would catch. Revisit when the
  service is built (M4): add a web search adapter behind the same
  protocol, and keep the snapshot for evals.
- **LLM judges can be wrong**, especially a small local one. Every
  verdict is saved with its reason for spot-checking. The first smoke
  run confirmed the risk: `qwen2.5:3b` as judge accepted facts the
  report got wrong (a wrong orbit, a wrong launch year), and its
  reasons were generic. Judge-based numbers are therefore only reported
  with a stronger judge (Cloudflare `llama-3.3-70b`) when quota allows;
  the code-based metrics stand on their own.
- **13 questions is a small sample**: one question moves "facts stated"
  by several points. The set is sized for CPU-only runs. Grow it before
  claiming small differences.
- **Quote matching is a lower bound** for "facts retrieved": a passage
  that states a fact in other words does not count.
