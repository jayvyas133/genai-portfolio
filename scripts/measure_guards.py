"""Measure how long the built-in guardrails take per call.

Runs every input guard on requests of several sizes, plus
the output guard on a reply of the same size, and prints
the median time over many runs. Guards cost microseconds
while a provider call costs hundreds of milliseconds, so
they are measured directly rather than inside a real call.

Usage:
    uv run python scripts/measure_guards.py
"""

import statistics
import time

from aicore.guardrails import (
    InjectionHeuristic,
    MaxInputChars,
    PIIRedactor,
    SecretLeakGuard,
    apply_input_guards,
    apply_output_guards,
)
from aicore.models import LLMRequest, LLMResponse, Message, Usage

RUNS = 200
SENTENCE = "A vector database stores embeddings and finds the nearest ones quickly. "

INPUTS = {
    "short chat (~100 chars)": SENTENCE + "Mail me at jay@example.com.",
    "RAG prompt (~4k chars)": SENTENCE * 55,
    "large RAG (~40k chars)": SENTENCE * 550,
    "worst case: 40k digits": "1234 5678 " * 4_000,
}


def median_us(text: str) -> float:
    """Median microseconds for all guards on ``text``, over ``RUNS`` runs."""
    input_guards = (MaxInputChars(100_000), PIIRedactor(), InjectionHeuristic())
    output_guards = (SecretLeakGuard(),)
    request = LLMRequest(messages=[Message(role="user", content=text)])
    reply = LLMResponse(text=text, model="m", usage=Usage(input_tokens=1, output_tokens=1))

    samples = []
    for _ in range(RUNS):
        start = time.perf_counter_ns()
        guarded = apply_input_guards(input_guards, request)
        apply_output_guards(output_guards, guarded, reply)
        samples.append((time.perf_counter_ns() - start) / 1_000)
    return statistics.median(samples)


def main() -> None:
    """Print one row per input size."""
    for label, text in INPUTS.items():
        print(f"{label:26} {len(text):>7,} chars  {median_us(text):>10,.1f} us")


if __name__ == "__main__":
    main()
