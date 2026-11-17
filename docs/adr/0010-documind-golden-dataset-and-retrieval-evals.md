# ADR (Architecture Decision Record) 0010: DocuMind golden dataset and retrieval evals

- **Status:** accepted
- **Date:** 2026-11-17
- **Scope:** 01-documind

## Context

DocuMind is a RAG (Retrieval-Augmented Generation) service. M4 will try
dense, hybrid and reranked retrieval, and M5 adds generation. Without a
fixed set of questions and a number, "hybrid is better" would be an
opinion. The forces were:

- **Labels must survive re-chunking.** Chunk ids changed once already in
  M2 (300 → 200 words) and will change again if chunking is tuned.
- **Retrieval and generation fail differently**, so they must be scored
  separately: a wrong answer from the right passage is a generation bug;
  a right passage never retrieved is a retrieval bug.
- **The sample is small** (tens of questions), so one question moves a
  percentage by several points.
- **Tuning on the same questions you report overfits** the numbers.

## Options considered

### How to label relevance
1. **Chunk ids.** Exact, but every chunking change silently invalidates
   the whole dataset.
2. **Document title only.** Robust, but a long document (AMD has 78
   chunks) would count any of its chunks as a hit.
3. **Document title plus a short verbatim quote** (chosen). A chunk is
   relevant when it is from that document and contains the quote. Survives
   re-chunking and is precise.

### Who writes the questions
1. **An LLM (Large Language Model) generates them from passages.** Cheap
   to repeat, but questions tend to copy the passage's wording, which
   flatters retrieval.
2. **Drafted by hand from the corpus, then a human reviews a sample**
   (chosen). Questions are reworded away from the source text; a validator
   proves every quote exists.

### Where the metrics live
1. **`aicore.evals`.** Its scorers judge one output pass/fail; ranking
   metrics need the whole ranked list.
2. **`documind.metrics`** (chosen). Retrieval metrics are pure functions
   over ranks. Generation evals in M5 will use `aicore.evals`.

## Decision

- **60 cases in `evals/golden.jsonl`**: 40 dev and 20 test. Each has a
  `kind`: 44 factual, 10 multi-hop (needs two passages from different
  documents) and 6 unanswerable (two of them near-misses that share words
  with the corpus). All 30 documents are covered, including deep sections,
  not only introductions.
- **Quotes are at most 30 words**, so they always fit inside one chunk
  (neighbouring chunks share 40 words). `validate_quotes` checks every
  quote against the actual chunks, not the raw text, so a quote in a
  dropped section or across a heading fails. It runs in the unit tests
  with the default chunk settings: a chunking change that breaks a label
  fails the build instead of quietly lowering scores. No quote matches
  more than two chunks, so no label is so generic it hits by luck.
- **Per case, only the rank of each evidence passage is stored**
  (`evidence_ranks`, e.g. `[5, 2]`). Hit rate@k, recall@k and MRR (Mean
  Reciprocal Rank) at any k are derived from it.
- **Unanswerable cases are excluded from retrieval metrics.** They have
  nothing to find and are scored on refusal in M5.
- **Evidence may list `alternatives`**: other documents stating the same
  fact. Added only after reviewing a miss and finding retrieval had
  returned the fact from an unlisted document.
- **Dev for tuning, test once.** The test split is not run until M4's
  final configuration is chosen.
- **Results are saved to `evals/results/{retriever}-{split}.json`** under
  a fixed name, with the settings that produced them. Committing the file
  makes `git diff` show which cases improved or regressed.

## Results (dense baseline, dev split)

BGE (BAAI General Embedding) `bge-base-en-v1.5` query embedding, pgvector
cosine top 10 over 888 chunks.

| Metric | All (36) | Factual (30) | Multi-hop (6) |
|---|---|---|---|
| Hit rate@1 | 0.67 (24/36) | 0.73 (22/30) | 0.33 (2/6) |
| Hit rate@5 | 0.94 (34/36) | 0.97 (29/30) | 0.83 (5/6) |
| Recall@10 | 0.93 | 0.97 | 0.75 |
| MRR@10 | 0.79 | 0.83 | 0.58 |

Latency per query (embed + search): p50 180 ms, p95 304 ms.

**Reviewing the misses** changed one label. "How much did AMD pay to
acquire ATI?" was scored a miss, but the chunk at rank 2 (AMD > History)
states "approximately $5.4 billion"; the label listed only the Radeon
document. That incomplete label was fixed with an alternative. The other
misses are genuine: a fact buried in a long specification list (Ampere),
and multi-hop questions where one document dominates the top results.

## Trade-offs

- **Small sample.** With 36 scored dev cases one case is about 3 points;
  differences under ~2 cases are noise. Raw counts are shown next to
  every rate. Confidence intervals (bootstrap) are deferred.
- **One author.** Questions were drafted by one person and 15 were
  reviewed by a second; a fully independent set would be stronger.
- **Labels can still be incomplete.** Only misses were reviewed. A case
  that hit at rank 4 might have had an unlisted relevant chunk at rank 1,
  which would understate MRR slightly.
- **Quote matching is literal.** A chunk that paraphrases the fact does
  not count. Short, distinctive quotes and alternatives reduce this.
- **No nDCG (normalized Discounted Cumulative Gain)**: with binary
  relevance and one or two passages per case, MRR and recall carry the
  same information.
