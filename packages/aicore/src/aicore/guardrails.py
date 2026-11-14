"""Guardrails: deterministic checks around every LLM call.

Input guards run before the cache and the provider; they
may return a changed request (e.g. with PII, Personally
Identifiable Information, redacted) or raise
``GuardrailError``. Output guards run on the reply before it
is cached or returned; they can only allow or block it.
"""

import re
import unicodedata
from collections.abc import Sequence
from typing import Protocol

from aicore.errors import GuardrailError
from aicore.models import LLMRequest, LLMResponse


class InputGuardrail(Protocol):
    """Checks, and may rewrite, a request before it is sent."""

    name: str

    def check(self, request: LLMRequest) -> LLMRequest:
        """Return the request to send, possibly changed.

        Raises:
            GuardrailError: If the request must not be sent.
        """
        ...


class OutputGuardrail(Protocol):
    """Checks a reply before it reaches the caller."""

    name: str

    def check(self, request: LLMRequest, response: LLMResponse) -> None:
        """Return normally if the reply is allowed.

        Raises:
            GuardrailError: If the reply must not be returned.
        """
        ...


def apply_input_guards(guards: Sequence[InputGuardrail], request: LLMRequest) -> LLMRequest:
    """Run ``guards`` in order, each on the previous one's output.

    Args:
        guards: Input guards, in the order they should run.
        request: The request as the caller built it.

    Returns:
        The request after every guard has run.

    Raises:
        GuardrailError: From the first guard that rejects it.
    """
    for guard in guards:
        request = guard.check(request)
    return request


def apply_output_guards(
    guards: Sequence[OutputGuardrail], request: LLMRequest, response: LLMResponse
) -> None:
    """Run every output guard; the first rejection wins.

    Raises:
        GuardrailError: From the first guard that rejects the reply.
    """
    for guard in guards:
        guard.check(request, response)


class MaxInputChars:
    """Block requests whose messages are too long in total.

    Characters, not tokens: no tokenizer dependency and the
    same behaviour for every provider. As a rough rule,
    English text is about 4 characters per token.
    """

    name = "max_input_chars"

    def __init__(self, max_chars: int) -> None:
        """Set the limit.

        Args:
            max_chars: Largest allowed total length of all
                message contents; must be > 0.
        """
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self.max_chars = max_chars

    def check(self, request: LLMRequest) -> LLMRequest:
        """Return ``request`` unchanged if it fits the limit.

        Raises:
            GuardrailError: If the total length exceeds ``max_chars``.
        """
        total = sum(len(m.content) for m in request.messages)
        if total > self.max_chars:
            raise GuardrailError(self.name, f"input is {total} chars, limit {self.max_chars}")
        return request


_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# 13-19 digits, optionally split by spaces or hyphens.
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
# Optional +country code, then exactly 10 digits (India, US, ...).
_PHONE = re.compile(r"(?<![\d+])(?:\+\d{1,3}[ -]?)?(?:\d[ -]?){9}\d(?!\d)")


def _luhn_ok(digits: str) -> bool:
    """Return True if ``digits`` passes the Luhn checksum.

    From the right, double every second digit (subtract 9
    if the result is over 9) and sum everything; real card
    numbers give a multiple of 10.
    """
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _redact_card(match: re.Match[str]) -> str:
    """Replace a digit run with ``[CARD]`` only if it is Luhn-valid."""
    digits = re.sub(r"\D", "", match.group())
    return "[CARD]" if _luhn_ok(digits) else match.group()


def redact_pii(text: str) -> str:
    """Replace emails, card numbers and phone numbers in ``text``.

    Regex-based: fast and dependency-free, but it misses
    some formats (e.g. ``(415) 555-0123``) and can flag a
    10-digit number that is not a phone. Order matters:
    emails first (they may contain digits), then cards
    (which contain phone-like runs), then phones.
    """
    text = _EMAIL.sub("[EMAIL]", text)
    text = _CARD.sub(_redact_card, text)
    return _PHONE.sub("[PHONE]", text)


class PIIRedactor:
    """Redact PII (Personally Identifiable Information) from requests.

    User and assistant messages are redacted; system
    messages are written by the developer and kept as is.
    """

    name = "pii_redactor"

    def check(self, request: LLMRequest) -> LLMRequest:
        """Return ``request``, or a redacted copy if PII was found.

        The original request is never modified.
        """
        messages = []
        changed = False
        for message in request.messages:
            if message.role == "system":
                messages.append(message)
                continue
            redacted = redact_pii(message.content)
            if redacted != message.content:
                changed = True
                message = message.model_copy(update={"content": redacted})
            messages.append(message)
        if not changed:
            return request
        return request.model_copy(update={"messages": messages})


# (label, regex). Bounded gaps (.{0,20}) allow a few filler
# words but never match across a whole document.
_INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "ignore_previous",
        r"\b(?:ignore|disregard|forget)\b.{0,20}"
        r"\b(?:previous|prior|above|earlier)\b.{0,20}"
        r"\b(?:instructions|prompts?|rules|messages)\b",
    ),
    (
        "reveal_system_prompt",
        r"\b(?:reveal|show|print|repeat|leak)\b.{0,20}"
        r"\b(?:system prompt|hidden instructions|initial instructions)\b",
    ),
    (
        "role_override",
        r"\byou are now\b.{0,20}\b(?:dan|developer mode|jailbroken|unrestricted)\b",
    ),
)


def _normalize(text: str) -> str:
    """Fold look-alike Unicode and collapse whitespace.

    NFKC (Normalization Form Compatibility Composition)
    turns e.g. full-width letters into plain ASCII, so
    trivial disguises do not slip past the patterns.
    """
    return " ".join(unicodedata.normalize("NFKC", text).split())


class InjectionHeuristic:
    """Block user messages containing known prompt-injection phrases.

    A cheap first filter, not a defence: paraphrases, other
    languages and encoded text get through. Real protection
    is architectural (least privilege, validating what the
    model's output is allowed to do).
    """

    name = "injection_heuristic"

    def __init__(self, extra_patterns: Sequence[tuple[str, str]] = ()) -> None:
        """Compile the built-in patterns plus any extra ones.

        Args:
            extra_patterns: ``(label, regex)`` pairs a project adds.
                Matching is case-insensitive.
        """
        self._patterns = [
            (label, re.compile(pattern, re.IGNORECASE))
            for label, pattern in (*_INJECTION_PATTERNS, *extra_patterns)
        ]

    def check(self, request: LLMRequest) -> LLMRequest:
        """Return ``request`` unchanged if no user message matches.

        Raises:
            GuardrailError: Naming the first pattern that matched.
        """
        for message in request.messages:
            if message.role != "user":
                continue
            text = _normalize(message.content)
            for label, pattern in self._patterns:
                if pattern.search(text):
                    raise GuardrailError(self.name, f"matched {label}")
        return request


# (label, regex) for well-known credential formats.
_SECRET_PATTERNS: tuple[tuple[str, str], ...] = (
    ("aws_access_key", r"\bAKIA[0-9A-Z]{16}\b"),
    ("github_token", r"\bgh[pousr]_[A-Za-z0-9]{36,}"),
    ("api_key_sk", r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}"),
    ("google_api_key", r"\bAIza[0-9A-Za-z_-]{35}"),
    ("slack_token", r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
    ("private_key", r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    ("jwt", r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
)


class SecretLeakGuard:
    """Block replies that contain something that looks like a credential.

    Matches vendor prefixes (``AKIA``, ``ghp_``, ``sk-``, ...)
    rather than "random-looking" strings, so hashes and IDs
    in normal answers are not flagged.
    """

    name = "secret_leak"

    def __init__(self, extra_patterns: Sequence[tuple[str, str]] = ()) -> None:
        """Compile the built-in patterns plus any extra ones.

        Args:
            extra_patterns: ``(label, regex)`` pairs a project adds.
                Matching is case-sensitive: prefixes are exact.
        """
        self._patterns = [
            (label, re.compile(pattern)) for label, pattern in (*_SECRET_PATTERNS, *extra_patterns)
        ]

    def check(self, request: LLMRequest, response: LLMResponse) -> None:
        """Return normally if the reply has no credential-like text.

        Raises:
            GuardrailError: Naming the first pattern that matched.
        """
        for label, pattern in self._patterns:
            if pattern.search(response.text):
                raise GuardrailError(self.name, f"reply contains {label}")
