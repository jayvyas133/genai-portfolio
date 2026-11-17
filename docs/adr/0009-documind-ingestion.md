# ADR (Architecture Decision Record) 0009: DocuMind ingestion pipeline

- **Status:** accepted
- **Date:** 2026-11-17
- **Scope:** 01-documind

## Context

DocuMind is a RAG (Retrieval-Augmented Generation) service. Before it
can answer anything, its corpus must be split into chunks, embedded and
stored for search. The corpus is 30 plain-text articles about GPUs
(Graphics Processing Units): 105k words, with `== Heading ==`
section markers. The forces were:

- **The embedding model has a hard input limit.** BGE (BAAI General
  Embedding) `bge-base-en-v1.5` reads at most 512 tokens and silently
  drops the rest.
- **Embedding calls cost money and time**, and runs can crash part way
  (rate limits, network errors).
- **M4 needs both vector and keyword search** over the same chunks.
- **psycopg's async mode cannot use Windows' default Proactor event
  loop**, and development happens on Windows.

## Options considered

### Chunking
1. **Fixed token windows over the whole text.** Simple, but chunks
   straddle sections, so one vector mixes unrelated topics.
2. **Semantic chunking**, which splits where embedding similarity
   drops. Better boundaries, but it costs an embedding pass just to
   decide where to split, and the gain is hard to show without evals.
3. **Section-aware word windows** (chosen). Split on headings first,
   then slide a window of words with overlap inside each section.

### Re-ingesting a changed document
1. **Upsert chunks by (document, ordinal).** One edit early in a
   document shifts every later ordinal, so stale rows linger.
2. **Delete and re-insert the document's chunks in one transaction**
   (chosen). Always exact, and readers never see a half-written document.

### Avoiding repeat embedding
1. **Re-embed everything on every run.** Simple, but slow and costly.
2. **Content-hash reuse** (chosen). Each chunk's sha256 hash covers the
   exact text sent to the model. Vectors already stored under that hash
   are reused, across all documents.

## Decision

- **Chunking: sections first, then 200-word windows with 40 words of
  overlap.** A chunk never crosses a heading. Boilerplate sections
  (References, See also, External links and similar) are dropped.
  Overlap means a sentence cut at a window edge still appears whole in
  the neighbouring chunk.
- **The embedded text carries a header**: `Title > Section > Subsection`,
  then the chunk. A chunk that says "it was released in 2007" is
  embedded knowing it belongs to `CUDA > History`.
- **200 words was measured, not guessed.** The first version used 300
  words. Embedding the longest chunks one per call showed 3 chunks at
  exactly 512 tokens, meaning they were truncated. Dense sections, such
  as product tables, reach about 1.9 tokens per word, and 2.6 for pure
  model-number lists. At 200 words the densest prose chunk measured 400
  tokens. One chunk, a bare list of GPU model numbers, still hits 512.
  Its truncated tail is repeated in the next chunk's overlap, so nothing
  is lost from the index.
- **Batching stays within one document**, at up to 50 texts per call.
  Each document is stored (committed) before the next one starts, so a
  crash loses at most the document in progress. Re-running resumes,
  because finished documents reuse every vector.
- **The vector size is checked before writing.** If the model returns
  vectors that don't match `DOCUMIND_EMBEDDING_DIMENSIONS`, ingestion
  fails on the first batch and nothing is stored.
- **Schema.** `chunks.embedding vector(768)` has an HNSW (Hierarchical
  Navigable Small World) index with cosine distance. `chunks.tsv` is a
  generated `tsvector` column with a GIN (Generalized Inverted Index)
  index, ready for M4's hybrid search. Postgres keeps `tsv` in sync, so
  no application code can forget to update it.
- **Every embedding call goes through aicore** (`build_client`), so
  calls are retried, traced and swappable. The project never imports a
  provider class.
- **Windows:** the CLI (Command Line Interface) and the tests run on
  `asyncio.SelectorEventLoop` on Windows (`asyncio.run(...,
  loop_factory=...)` and pytest-asyncio's loop-factory hook).

## Results

| Run | Chunks | Embedded | Reused | Embed calls |
|---|---|---|---|---|
| First ingest, 300 words | 731 | 731 | 0 | 33 |
| Re-run, unchanged | 731 | 0 | 731 | 0 |
| Switch to 200 words | 888 | 465 | 423 | 30 |

Switching chunk size reused 423 vectors: short sections came out
identical, so their hashes matched.

## Trade-offs

- **Words are a proxy for tokens.** Word windows need no tokenizer
  dependency, but the safe size had to be measured. A new corpus with
  denser text (code, tables) should be re-measured.
- **Cloudflare bills padded batches.** A batch's token count is every
  text times the longest one (a 7-word chunk billed as 390 tokens next
  to a long chunk). Sorting texts by length before batching would cut
  billed tokens. Not worth it at this corpus size and on the free tier.
- **Tiny chunks are kept.** 43 chunks are under 20 words, mostly table
  residue. Filtering them is left until M3's evals can show whether they
  hurt retrieval.
- **Delete and re-insert rewrites a whole document's rows** for a
  one-word edit. Fine for 30 documents; for millions of rows, diffing by
  hash would write less.
- **HNSW is approximate.** At 888 rows an exact scan would also be fast.
  The index is there so the design scales, and its recall is checked in
  M4.
