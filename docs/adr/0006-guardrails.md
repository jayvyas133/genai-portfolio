# ADR (Architecture Decision Record) 0006: Guardrails

- **Status:** accepted
- **Date:** 2026-11-14
- **Scope:** aicore

## Context

Every LLM (Large Language Model) call in the portfolio goes through
`LLMClient`. Some risks are the same for every project and are cheap to
reduce in one place:

- **PII (Personally Identifiable Information) leaving the process.**
  User text may contain emails, phone numbers or card numbers. Once
  sent, it sits in the provider's logs, in our traces and, since M3, in
  the Redis cache.
- **Oversized prompts.** A pasted log file makes a slow, expensive call
  that may fail on the context window anyway.
- **Prompt injection.** Users (or retrieved documents) try to override
  the system prompt.
- **Secrets in replies.** A model can repeat a key from its prompt, from
  a retrieved document or from its training data.

Rules that depend on one project (for example, "only `SELECT`
statements" in the Text-to-SQL agent) do not belong here.

## Options considered

1. **A guardrails framework (NeMo Guardrails, Guardrails AI).**
   Pro: many ready-made validators.
   Con: large dependencies, their own configuration languages, and
   several validators call an LLM, which adds latency and cost per call.
   Against the portfolio rule of minimal, justified dependencies.
2. **LLM-as-judge checks** (a second model call classifies the input or
   output).
   Pro: catches paraphrases and other languages.
   Con: doubles latency and cost, and is itself an unmeasured model. It
   cannot be trusted until the evaluation harness (M4) can measure its
   false positive and false negative rates.
3. **Deterministic guards behind two small protocols, run inside
   `LLMClient`.**
   Pro: microseconds per call, no dependencies, fully unit-testable;
   same pattern as `LLMProvider`, `Tracer` and `ResponseCache`; projects
   add their own guards without changing aicore.
   Con: regexes miss what they were not written for.

## Decision

Option 3. Supporting details:

- **Two protocols.** `InputGuardrail.check(request) -> LLMRequest` may
  return a changed (redacted) request or raise.
  `OutputGuardrail.check(request, response) -> None` may only allow or
  block: silently rewriting a model's answer would hide bugs.
- **`GuardrailError(NonRetryableError)`** names the guard and gives a
  short, fixed reason. It never contains the offending text, because
  errors are logged and traced. Guards are deterministic, so retrying
  would be blocked again.
- **Order inside `complete()`:** resolve model, input guards, cache key,
  cache lookup, provider (with retries, only on a miss), output guards,
  cache write. Consequences:
  - The redacted request is what gets hashed, cached, sent and traced,
    so PII never reaches the provider or Redis.
  - A reply blocked by an output guard is never cached.
  - Output guards also run on cache hits, so a reply cached before a
    new pattern was added is still checked.
  - A blocked call still produces exactly one `CallRecord`
    (`outcome="GuardrailError"`). An output block keeps `usage` and
    `cost_usd`: the provider answered and billed before the guard ran.
- **Built-in guards:**
  - `MaxInputChars`: counts characters, not tokens, so no provider
    tokenizer is needed. Roughly 4 characters per token for English;
    Indic scripts use more tokens per character.
  - `PIIRedactor`: emails, phone numbers (optional country code plus 10
    digits) and card numbers, replaced by `[EMAIL]`, `[PHONE]`, `[CARD]`.
    Card candidates must pass the Luhn checksum, so most order IDs are
    left alone. System messages, written by the developer, are not
    redacted. Redacting rather than blocking keeps the call useful; the
    model rarely needs the actual address.
  - `InjectionHeuristic`: labelled patterns with bounded gaps
    (`.{0,20}`), case-insensitive, after NFKC (Normalization Form
    Compatibility Composition) and whitespace folding. Only user
    messages are scanned (retrieved documents are placed in the user
    message, so they are scanned too).
  - `SecretLeakGuard` (output): vendor prefixes (`AKIA`, `ghp_`, `sk-`,
    `AIza`, `xox?-`, private key headers, JWTs (JSON Web Tokens)) rather
    than "random-looking" strings, which would flag hashes and IDs.

## Measured overhead

`scripts/measure_guards.py`: all four guards on one request and its
reply, median of 200 runs, on 2026-11-14 on a CPU-only (Central
Processing Unit) Windows laptop:

| Input | Size | Time |
|---|---|---|
| Short chat message | 99 chars | 0.04 ms |
| RAG (Retrieval-Augmented Generation) prompt | 3,960 chars | 1.5 ms |
| Large RAG context | 39,600 chars | 14.6 ms |
| Worst case: digits and spaces | 40,000 chars | 33.0 ms |

Time grows linearly with input size (10× the text, 10× the time), also
for the worst case (0.2, 2.1, 20.3 ms for 400, 4,000, 40,000 chars of
digits through `redact_pii`). Every repeat in the patterns is bounded,
so there is no ReDoS (Regular expression Denial of Service). Against an
800 ms provider call, the overhead is 0.005% for chat and about 2% for a
10,000-token prompt. In the worst case most of the time is the
per-match Python callback that runs the Luhn check, not backtracking.

## Trade-offs

- **The injection heuristic is a speed bump, not a defence.**
  Paraphrases, other languages (including Hindi and other Indic
  languages), encodings such as base64, and indirect injection worded
  differently all get through. Real protection is architectural: least
  privilege (for example a read-only database user in project 02), and
  validating what the model's output is allowed to trigger.
- **Regex PII detection misses formats and names.** It misses
  `(415) 555-0123`, names, addresses and Aadhaar numbers, and it flags a
  10-digit Unix timestamp as a phone number. The upgrade path is
  Microsoft Presidio or an NER (Named Entity Recognition) model, behind
  the same `InputGuardrail` protocol.
- **Redaction changes the answer's input.** A question that depends on
  the actual email ("is this address valid?") cannot be answered. A
  project that needs that must not use `PIIRedactor`.
- **Redacted requests share cache entries.** Two users asking the same
  question with different emails get the same cached reply. This is
  correct, because the model never saw either email, but it is worth
  knowing.
- **`CallRecord` does not name the guard** that fired; only the outcome
  `GuardrailError` is recorded. The guard and reason are on the
  exception. A `guard` field can be added when a dashboard needs it.
- **Secrets in the input are not checked.** A user pasting their own key
  is sent as is; an input version of `SecretLeakGuard` is future work.
- Guards run on the event loop. At 15 ms for 40,000 characters this is
  acceptable; much larger inputs should be capped by `MaxInputChars`
  first, which is why it should be the first guard in the list.
