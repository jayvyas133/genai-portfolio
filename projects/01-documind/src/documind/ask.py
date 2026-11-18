"""Answer one question end to end: guard, retrieve, generate.

``answer_question`` is the one entry point the CLI (``documind ask``),
the answer eval and the HTTP API share, so all three run the same
pipeline with the same guards.
"""

import logging

from aicore.client import LLMClient
from aicore.errors import GuardrailError

from documind.generation import CONTEXT_CHUNKS, Answer, generate_answer
from documind.guards import QUESTION_MAX_CHARS, check_question
from documind.retrieval import Retriever

logger = logging.getLogger(__name__)

_REFUSAL_MESSAGES = {
    "no_sources": "No relevant passages were found in the documents.",
    "model": "The documents do not contain the answer to this question.",
    "uncited": "An answer was produced but could not be traced to a source, so it is not shown.",
}
_GUARD_MESSAGES = {
    "empty_question": "The question is empty.",
    "max_input_chars": f"The question is too long (limit {QUESTION_MAX_CHARS} characters).",
    "injection_heuristic": "The question looks like an attempt to override the instructions.",
    "secret_leak": "The answer was blocked because it appeared to contain a credential.",
}


async def answer_question(
    question: str,
    retriever: Retriever,
    client: LLMClient,
    *,
    k: int = CONTEXT_CHUNKS,
) -> Answer:
    """Check ``question``, retrieve ``k`` chunks and answer from them.

    A question rejected by a guard is returned as a ``blocked`` refusal
    without any retrieval or model call. A PII-redacted question is
    what gets retrieved with and sent to the model.

    Raises:
        LLMError: If an embedding, rerank or model call fails after retries.
    """
    try:
        checked = check_question(question)
    except GuardrailError as err:
        logger.info("Question blocked by %s", err.guard)
        return Answer(text="", refusal_reason="blocked", blocked_by=err.guard)
    chunks = await retriever.retrieve(checked, k)
    # Logged, not used as a gate yet: the scores show whether a
    # threshold would separate unanswerable questions (see ADR 0012).
    logger.info("Retrieved scores: %s", [round(c.score, 3) for c in chunks])
    return await generate_answer(checked, chunks, client)


def refusal_message(answer: Answer) -> str:
    """Return a short, user-facing explanation of why ``answer`` was refused.

    Never includes the question or the model's text, which may hold PII
    or the very content a guard blocked.
    """
    if answer.refusal_reason == "blocked":
        return _GUARD_MESSAGES.get(answer.blocked_by or "", "The request was blocked.")
    return _REFUSAL_MESSAGES.get(answer.refusal_reason or "", "No answer.")
