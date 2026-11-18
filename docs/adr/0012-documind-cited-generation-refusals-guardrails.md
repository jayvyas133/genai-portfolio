# ADR (Architecture Decision Record) 0012: DocuMind cited generation, refusals and guardrails

- **Status:** accepted
- **Date:** 2026-11-18
- **Scope:** 01-documind, aicore (`build_client` guard parameters,
  Cloudflare JSON-reply fix)

## Context

M4 delivered retrieval that ranks an answer passage in the top 5 for
36 of 36 dev questions. M5 turns those chunks into an answer. The
forces were:

- **Every shown answer must be traceable** to a chunk the user can read.
  An LLM (Large Language Model) can write fluent text that no source
  supports.
- **The system must say "I don't know"** when the documents do not hold
  the answer (6 unanswerable golden questions), without refusing
  questions it can answer.
- **Documents are untrusted input.** Their text goes into the prompt, so
  they can carry instructions. Questions can carry PII (Personally
  Identifiable Information) or injection attempts.
- **Quality claims need evals** with a judge whose mistakes are visible,
  on a small sample (36 dev, 18 test answerable questions).
- **Model:** `@cf/meta/llama-3.3-70b-instruct-fp8-fast` on Cloudflare
  Workers AI, at temperature 0.

## Options considered

### How the answer cites its sources
1. **A separate `citations: [1, 3]` list in the JSON reply.** The list
   and the text can disagree, and the reader never sees the list next
   to the claim.
2. **`[n]` markers in the answer text** (chosen). The chunks are
   numbered in the prompt and the model writes the numbers after each
   sentence. The markers *are* the citations: what the reader sees is
   what gets checked. Numbers outside `1..n` are reported as invalid,
   never shown as sources.

### How refusal works
1. **The model decides alone** (`refused: true`). Cheap, but a model that
   answers without citing anything gives no way to check the answer.
2. **A rerank-score threshold before the model call.** It saves a call,
   but a threshold tuned on 4 dev unanswerable questions would mostly
   fit noise. The scores are logged instead: answerable questions
   scored about 0.99, "capital of France" about 0.002.
3. **The model's flag plus a rule: an answer with no valid citation is
   withheld** (chosen). The refusal reasons are explicit:
   `no_sources` (nothing retrieved, so no model call), `model` (the
   model refused), `uncited` (answered without a valid marker; the text
   is kept as `draft` for evals, never shown) and `blocked` (a
   guardrail fired).

**Getting the model to cite.** Prompt v1 (rules only) left 2 of 36 dev
answers without markers, returned one non-JSON reply and refused one
answerable question. Prompt v2 adds one worked example: on the same
dev run, 35 of 36 were cited and there were 0 JSON errors. A reply that is plain
text instead of JSON is used as the answer if it carries valid markers
(`plain_text_reply=True`). The citation rule still applies, so an
uncited prose refusal is never shown.

### Where the guardrails run
1. **As input guards on the model request.** The request also contains
   the retrieved documents. PII redaction would rewrite source text, so
   a cited chunk would no longer match the stored one. The injection
   regex would also block an honest question whenever a source quotes
   an injection-like phrase.
2. **On the question alone, before retrieval** (chosen). `check_question`
   strips the question and runs aicore's `MaxInputChars(1000)`,
   `PIIRedactor` and `InjectionHeuristic`. A blocked question costs
   nothing (no retrieval or model call). The redacted question is what
   gets embedded, reranked, sent and logged, so PII never reaches a
   provider. A question over the limit is **refused, not truncated**:
   truncation could cut off the actual question.
3. **Documents** are wrapped in `<sources>` tags, and the system prompt
   says they are data, not instructions. That is a mitigation, not a
   guarantee. Real protection is that the model's output can't do
   anything: there are no tools and nothing is executed.
4. **Output:** `SecretLeakGuard` runs inside `LLMClient` (aicore's
   `build_client` gained `input_guards` and `output_guards`), so a
   blocked reply is traced like any other failed call and becomes a
   `blocked` refusal.

### How answers are scored
| Measure | Method | Why |
|---|---|---|
| Refusal accuracy | Unanswerable refused; answerable refused = false refusal | Bookkeeping, no model |
| Citation precision | Cited chunk contains a labelled evidence quote (same `is_relevant` rule as the retrieval eval) | Deterministic, free |
| Correctness | LLM judge vs `reference_answer`: correct / partial / incorrect, reason first | Wording varies too much for string matching |

The judge is the same model that answers. It grades against a written
reference, which limits self-preference bias. Every verdict is saved
with its reason and 10 were spot-checked by hand. Each saved result
records `PROMPT_VERSION` and `JUDGE_VERSION`.

## Decision

`[n]` markers as citations, enforced in code: no valid citation means
no answer. Guardrails run on the question before retrieval, and the
secret-leak guard runs on the reply. One `answer_question` function is
shared by `documind ask`, `documind eval-answers` and the M6 API.

## Results

hybrid-rerank, top 5 chunks, prompt v2, judge v1. Dev was used to build
the prompt. Test was run once, with no changes afterwards.

| Metric | Dev (36 answerable + 4 unanswerable) | Test (18 + 2) |
|---|---|---|
| Unanswerable refused | 4/4 | 2/2 |
| False refusals | 0/36 | 1/18 (multi-008) |
| Answers with a supported citation | 35/36 | 17/17 |
| Citation precision (lower bound) | 0.69 (38/55) | 0.57 (21/37) |
| Judge: correct, end to end | 33/36 (factual 30/30, multi-hop 3/6) | 15/18 (factual 12/14, multi-hop 3/4) |
| Latency per answer p50 / p95 | 2.1 s / 2.8 s | 2.0 s / 2.7 s |
| Cost per answer | $0.00055 | $0.00054 |

Reading the misses:

- **Multi-hop misses are mostly retrieval.** `multi-004` and `multi-005`
  (dev) and `multi-008` (test) never got their second passage into the
  top 5. The model then answered part, refused, or (in `multi-004`)
  gave the IPO date instead of the founding date.
- **Two of the three non-correct test verdicts are judge errors.**
  `shader-002` was marked partial for adding a true detail, which the
  judge prompt allows. `pcie-002` was marked partial for omitting a
  detail the question did not ask for. By hand, test is 17/18 correct.
  The judge was not changed after seeing test results.
- **Citation precision is a lower bound.** The model often cites 3–4
  chunks per sentence. Some extra citations are wrong (an IPO chunk
  cited for a founding date), others state the same fact in a chunk the
  evidence label does not list.
- **A faithfulness gap the checks can't see:** in `multi-003` the model
  wrote the company headquarters from its own knowledge, said it was
  not in the sources, and still attached a citation.

## Consequences and trade-offs

- **Answers take about 2 s**: embed 0.2–0.3 s, rerank 0.5–1.3 s, model
  1–2 s. One run took 33 s on a slow Cloudflare response; timeouts and
  streaming belong to the API (M6).
- **Enforced citations can withhold a correct answer** that the model
  forgot to cite. With prompt v2 this happened 0 times on dev and test.
- **A partial answer is allowed** if it says what is missing. That
  matches the prompt rule but means "answered" is not "fully answered":
  read the judge rows, not the answered rate.
- **The judge shares the answerer's model** and is lenient on extra
  detail and strict on missing detail in unpredictable ways. Its
  numbers come with a hand-corrected count, not on their own.
- **Small sample:** one dev case is about 3 points, one test case about
  6 points. All runs are single runs at temperature 0, which is close
  to, but not exactly, deterministic: `multi-006` was refused in one dev
  run and answered correctly in the next.

## Future work

- Multi-hop: decompose the question into sub-questions, or retrieve
  more chunks for questions that name two things.
- A faithfulness judge (is every sentence supported by its cited
  chunk?) would catch outside knowledge like `multi-003`, and would
  measure true citation precision instead of a label-based lower bound.
- Discourage over-citation (cite the single best source per sentence).
- A rerank-score gate, once there are enough unanswerable questions to
  set it on dev and check it on test.
- Make `QUESTION_MAX_CHARS` a setting.
- Injection test cases planted inside documents.
