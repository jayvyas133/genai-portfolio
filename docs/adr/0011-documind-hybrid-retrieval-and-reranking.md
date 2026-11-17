# ADR (Architecture Decision Record) 0011: DocuMind hybrid retrieval and reranking

- **Status:** accepted
- **Date:** 2026-11-17
- **Scope:** 01-documind

## Context

The M3 dense baseline (BGE (BAAI General Embedding) `bge-base-en-v1.5`,
pgvector cosine) found an answer passage in the top 5 for 34 of 36 dev
questions, but ranked it first for only 24, and multi-hop questions were
weakest (2 of 6 at rank 1). M5 will pass the top few chunks to an LLM
(Large Language Model), so rank 1–3 is what matters. The forces were:

- **Embeddings blur exact tokens.** Model numbers and names ("RTX 3090",
  "HBM2e") can sit close to their neighbours in vector space.
- **A bi-encoder scores the question and each chunk separately**, so it
  cannot check whether a chunk actually answers the question.
- **Latency and cost per query**: every extra model call adds both.
- **The sample is small** (36 dev, 18 test answerable cases): one case
  is about 3 points on dev and 6 on test.

## Options considered

### Keyword search
1. **Postgres full-text search on the existing `tsv` column** (chosen).
   No new service; the GIN (Generalized Inverted Index) index was built
   in M2. Its ranking (`ts_rank_cd`) has no IDF (Inverse Document
   Frequency): a common word counts as much as a rare one.
2. **BM25 (Best Matching 25) via an extension or a search engine.**
   Better ranking, but a new dependency or service to run.

**AND or OR between the question's words.** `websearch_to_tsquery`
requires every word, so a natural-language question matches almost
nothing. The question is turned into an OR query instead: stemmed,
stop words dropped, any word may match, more matches rank higher.

### Merging dense and keyword results
1. **Add or weight the scores.** Cosine similarity (−1 to 1) and
   `ts_rank_cd` (0 upwards, unbounded) are on different scales, so one
   list would dominate.
2. **RRF (Reciprocal Rank Fusion)** (chosen). Each chunk scores
   Σ 1 / (60 + rank) over the lists it appears in. Uses ranks only, has
   one constant (60, from the original paper) and no training.

### Reranking
1. **None.**
2. **Cross-encoder reranker `bge-reranker-base` through `aicore`**
   (chosen). Reads question and chunk together; applied to the top 30
   candidates only, because it is slower than vector search.
3. **An LLM as reranker.** More flexible, but slower and far costlier
   per query.

Retrievers compose: `HybridRetriever(dense, sparse)` and
`RerankingRetriever(inner)` wrap any `Retriever`, so five configurations
were measured with the same runner: dense, sparse, hybrid, dense-rerank
and hybrid-rerank.

## Results

Top 10 of 888 chunks. Hybrid fetches 50 candidates per search; the
reranker reranks 30. Latency is per query from the CLI (Command Line
Interface) on a home connection to Cloudflare Workers AI, so it varies
between runs: dense p50 moved between 170 and 240 ms.

**Dev (36 answerable cases, used to choose):**

| Retriever | Hit@1 | Hit@3 | Hit@5 | Recall@10 | MRR@10 | p50 ms |
|---|---|---|---|---|---|---|
| dense (baseline) | 24/36 | 32/36 | 34/36 | 0.93 | 0.79 | 240 |
| sparse | 9/36 | 22/36 | 26/36 | 0.74 | 0.44 | 7 |
| hybrid | 20/36 | 30/36 | 30/36 | 0.90 | 0.69 | 279 |
| dense-rerank | 32/36 | 35/36 | 35/36 | 0.96 | 0.93 | 824 |
| **hybrid-rerank** | **32/36** | **36/36** | **36/36** | **0.96** | **0.94** | 922 |

Multi-hop at rank 1: dense 2/6, hybrid 3/6, both rerankers 5/6.

**Test (18 answerable cases, run once after choosing, nothing tuned after):**

| Retriever | Hit@1 | Hit@3 | Hit@5 | Recall@10 | MRR@10 | p50 ms |
|---|---|---|---|---|---|---|
| dense (baseline) | 15/18 | 16/18 | 17/18 | 0.89 | 0.86 | 170 |
| sparse | 7/18 | 10/18 | 13/18 | 0.86 | 0.51 | 7 |
| hybrid | 10/18 | 15/18 | 15/18 | 0.81 | 0.67 | 252 |
| dense-rerank | 16/18 | 17/18 | 18/18 | 0.97 | 0.93 | 865 |
| **hybrid-rerank** | **16/18** | **17/18** | **17/18** | **0.97** | **0.93** | 849 |

**Reading the results:**

- **Reranking is the real gain**: MRR (Mean Reciprocal Rank) 0.79 → 0.94
  on dev, 0.86 → 0.93 on test. It also found gpgpu-002, which every
  first stage ranked low or missed.
- **Hybrid without a reranker is worse than dense** (dev Hit@1 24 → 20,
  test 15 → 10). Keyword search alone is weak here (dev MRR 0.44):
  without IDF, "Which company created OpenGL" ranks any chunk with
  "OpenGL" and "company" highly, and document titles are not in the
  keyword index. RRF weights both lists equally, so the noisy list
  pushed correct dense results down (amd-001 fell from rank 1 to 10).
- **dense-rerank and hybrid-rerank are tied within noise.** On dev each
  wins one case: keyword search matched "RTX 3090" exactly and rescued
  multi-005; dense-rerank found both passages of multi-006.

## Decision

- **Default retriever: `hybrid-rerank`.** Same quality as dense-rerank
  within noise. The keyword leg costs about 10 ms and covers exact names
  and model numbers, which real users type and the golden set
  under-samples.
- **Not hybrid alone.** Measured as worse than dense on both splits.
- **Settings fixed at the defaults**: RRF constant 60, 50 candidates per
  search, 30 reranked. They were not swept: with 36 dev cases a sweep
  would mostly fit noise.
- **Hybrid runs its two searches one after the other**, not
  concurrently: they share one Postgres connection, which runs one query
  at a time, and keyword search takes milliseconds next to the
  embedding call.
- **Chunks reach the reranker with their `Title > Section` header**, the
  same text that was embedded (`chunking.with_header`), so both models
  see the same context.

## Trade-offs

- **Latency roughly quadruples**: p50 about 200 ms → 850–920 ms, p95 up
  to 1.8 s. Acceptable for a question-answering service where the LLM
  call that follows (M5) takes seconds; not for search-as-you-type.
- **One more paid call per query** (the rerank call) and one more
  provider dependency on the request path. A fallback to the un-reranked
  order if it fails is deferred to the API milestone.
- **Postgres keyword ranking has no IDF.** BM25 (for example with an
  extension such as ParadeDB's `pg_search`) or adding document titles to
  the `tsv` column might make hybrid help on its own; not done, because
  the reranker already recovers the ranking.
- **The reranker reads at most 512 tokens per question + chunk.** Chunks
  are at most about 400 tokens (ADR 0009), so nothing is cut today; a
  larger chunk size would need re-measuring.
- **Small samples**: the dense-rerank vs hybrid-rerank choice rests on
  judgement, not on a measured difference.
