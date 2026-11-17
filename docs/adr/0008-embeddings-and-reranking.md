# ADR (Architecture Decision Record) 0008: Embeddings and reranking in aicore

- **Status:** accepted
- **Date:** 2026-11-17
- **Scope:** aicore (needed by 01-documind)

## Context

DocuMind, project 01, is a RAG (Retrieval-Augmented Generation)
service. It needs two kinds of model call besides chat completion:

- **Embedding:** turn document chunks and user queries into vectors
  for pgvector similarity search.
- **Reranking:** a cross-encoder rescores the top candidates from
  retrieval before they go into the prompt.

The portfolio rule is that every LLM (Large Language Model), embedding
and reranker call goes through aicore, so these calls get the same
timeouts, retries, tracing and cost tracking as `complete()`. Two facts
shaped the design:

- **Not every backend offers every capability.** Ollama can embed but
  has no rerank endpoint; Cloudflare Workers AI offers all three.
- **Retrieval embedding models are asymmetric.** BGE (BAAI General
  Embedding) expects short queries to carry an instruction prefix
  ("Represent this sentence for searching relevant passages: "), and
  documents to carry none. Other models, such as nomic, use different
  prefixes. Callers should not need to know these.

## Options considered

1. **Add `embed` and `rerank` methods to `LLMProvider`.**
   - Pro: one protocol, one object to pass around.
   - Con: every adapter must implement all three methods. Ollama would
     need a `rerank` stub that raises `NotImplementedError`, so a wiring
     mistake that mypy could catch would instead crash at runtime.
2. **Separate protocols per capability, used by separate client classes
   (`EmbeddingClient`, `RerankClient`).**
   - Pro: each class is small.
   - Con: the timeout, retry and tracing logic would be copied, or
     moved into a shared base class. Projects would build three clients.
3. **Separate protocols (`EmbeddingProvider`, `RerankProvider`), with
   optional `embedder=` and `reranker=` arguments on the existing
   `LLMClient`.**
   - Pro: one client and one retry loop. mypy checks each capability on
     its own. A client without a reranker type-checks, and calling
     `rerank()` on it raises a clear `InvalidRequestError`.
   - Con: `LLMClient` grows two arguments and two methods. Unusable
     methods are only detected when called.

## Decision

Option 3. Supporting details:

- **Models** (`models.py`):
  - `EmbedRequest(texts, input_type, model)`. It takes a batch of texts,
    because ingestion embeds thousands of chunks and one HTTP (Hypertext
    Transfer Protocol) call per chunk would be far slower.
  - `input_type` is `"query"` or `"document"` (default `"document"`).
    The adapter turns it into the model's own prefix, so callers state
    intent, not model-specific strings.
  - Empty batches and empty strings are rejected at validation time.
  - `RerankRequest(query, documents, top_n, model)` returns
    `RerankResult(index, score)`. Results point back into the request by
    index rather than echoing the text, because the caller already holds
    the chunks and their metadata.
  - The existing `Usage` model is reused, with `output_tokens=0`.
- **Protocols** (`providers/base.py`): `EmbeddingProvider.embed` and
  `RerankProvider.rerank`. `FakeProvider` and `CloudflareProvider`
  satisfy all three protocols; `OllamaProvider` does not change yet.
- **One retry loop** (`client.py`):
  - `complete()`, `embed()` and `rerank()` share `_with_retries()`.
  - It takes a factory (`Callable[[], Awaitable[T]]`), not a coroutine,
    because a coroutine can be awaited only once and each attempt needs
    a fresh one.
  - A small mutable `_Attempts` counter lets the caller trace the number
    of attempts even when the loop raises.
- **Tracing:** `CallRecord` gains `operation` (`complete`, `embed` or
  `rerank`; default `complete`, so older records stay valid). Cost and
  latency can then be split by call type.
- **Default models:** `AICORE_EMBED_MODEL` and `AICORE_RERANK_MODEL`.
- **Offline fake** (`providers/fakeai.py`):
  - Embedding uses the hashing trick. Each word is hashed with
    `blake2b` into a signed bucket, and the vector is L2-normalised.
    Texts that share words get a high cosine similarity, so offline
    retrieval tests produce meaningful rankings.
  - `blake2b` is used instead of `hash()` because Python salts string
    hashes per process, which would change vectors between runs.
  - Reranking scores the fraction of query words that appear in each
    document.
  - `embed_errors` and `rerank_errors` script failures for retry tests.
- **Cloudflare** (`providers/cloudflare.py`):
  - Embedding uses `@cf/baai/bge-base-en-v1.5` (768 dimensions) and
    reranking uses `@cf/baai/bge-reranker-base`.
  - The three endpoints share one `_run()`, so they classify errors
    identically.
  - The rerank results are re-sorted and truncated to `top_n` on our
    side, so the contract holds even if the API's ordering changes.

## Verified against the live API (2026-11-17)

Using the integration test and a raw request:

- **Embedding** returns `{shape, data, pooling, usage}`, and `usage`
  holds `prompt_tokens`. It reports no neurons.
- **Rerank** returns `{response: [{id, score}], meta}` with no usage.
- **Rerank scores are raw and small.** For "gpu memory", an H100 memory
  sentence scored 0.049 and "cricket" scored 0.032. The order is right,
  but the absolute values are not calibrated probabilities, so DocuMind
  must not apply a fixed score threshold without measuring one on its
  eval set.
- **Single-call latency:** about 700 ms for a cold embedding call of 2
  texts and about 235 ms for a rerank of 2 documents. This is one
  sample, an order of magnitude only.

## Trade-offs

- **Cost is unknown for Cloudflare embed and rerank.** Neither endpoint
  reports neurons, so `cost_usd` is `null`. It is never reported as
  free. Estimating neurons from a per-model token price is deferred.
- **No embedding cache yet.** Re-ingesting unchanged chunks re-embeds
  them. A content-hash cache is cheap to add when M2 ingestion shows
  repeat work.
- **No batch splitting.** Cloudflare limits the batch size per call;
  ingestion batches its chunks itself.
- **Guardrails do not run on `embed`/`rerank`.** PII (Personally
  Identifiable Information) redaction before embedding is a
  per-project decision, because redacted text embeds differently.
- **`build_provider` still returns only an `LLMProvider`.** DocuMind
  passes `embedder=` and `reranker=` explicitly; a factory for them can
  follow if a second project needs it.
- **Ollama embeddings are deferred.** The protocol is ready; the
  adapter method is a small follow-up.
