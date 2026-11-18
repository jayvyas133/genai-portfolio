"""Guardrails: checks on the question going in and the answer coming out.

The question is checked on its own, before retrieval, not as part of the
model request. The request also carries the retrieved documents, and
running the input guards on them would go wrong in two ways:

- PII (Personally Identifiable Information) redaction would rewrite
  source text, so a cited chunk would no longer match what is stored.
- The injection check would block an honest question whenever a source
  happens to quote an injection-like phrase.

Checking first also means a redacted question is what gets embedded,
reranked and logged, so PII never reaches any provider. Documents are
handled differently: they are wrapped in ``<sources>`` tags and the
system prompt says they are data, not instructions (see
``generation.SYSTEM_PROMPT``).

The answer is checked by ``SecretLeakGuard`` inside ``LLMClient``, so a
blocked reply is traced like any other failed call.
"""

from aicore.errors import GuardrailError
from aicore.guardrails import (
    InjectionHeuristic,
    InputGuardrail,
    MaxInputChars,
    OutputGuardrail,
    PIIRedactor,
    SecretLeakGuard,
    apply_input_guards,
)
from aicore.models import LLMRequest, Message

QUESTION_MAX_CHARS = 1000
"""The longest golden question is 104 characters. A 1,000-character
limit leaves plenty of room for real questions while stopping pasted
documents and padding attacks."""


def question_guards() -> tuple[InputGuardrail, ...]:
    """Return the guards run on every question, in order.

    Length first because it is the cheapest check; redaction before the
    injection check so the error path never handles raw PII.
    """
    return (MaxInputChars(QUESTION_MAX_CHARS), PIIRedactor(), InjectionHeuristic())


def answer_guards() -> tuple[OutputGuardrail, ...]:
    """Return the guards run on every model reply."""
    return (SecretLeakGuard(),)


def check_question(question: str) -> str:
    """Return ``question`` stripped and with PII redacted, or raise if it is blocked.

    Raises:
        GuardrailError: If the question is empty, too long, or looks like
            a prompt injection. ``err.guard`` names the check.
    """
    question = question.strip()
    if not question:
        raise GuardrailError("empty_question", "question is empty")
    request = LLMRequest(messages=[Message(role="user", content=question)])
    checked = apply_input_guards(question_guards(), request)
    return checked.messages[0].content
