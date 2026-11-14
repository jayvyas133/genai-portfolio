"""Exception types raised by aicore's LLM client and provider adapters.

Adapters translate vendor-specific errors into these classes, so the client
can decide whether to retry by checking one base class (``RetryableError``)
instead of knowing every vendor's exceptions.
"""


class LLMError(Exception):
    """Base class for every error raised by aicore's LLM layer."""


class RetryableError(LLMError):
    """A transient failure; the same request may succeed if sent again."""


class NonRetryableError(LLMError):
    """A permanent failure; resending the same request will fail again."""


class RateLimitError(RetryableError):
    """The provider rejected the call for exceeding a quota (HTTP 429)."""

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        """Store the provider's suggested wait time, if it sent one.

        Args:
            message: Human-readable description of the failure.
            retry_after_s: Seconds the provider asked us to wait, taken from
                its ``Retry-After`` header; ``None`` if not provided.
        """
        super().__init__(message)
        self.retry_after_s = retry_after_s


class ProviderTimeoutError(RetryableError):
    """The call did not finish within the configured timeout."""


class ProviderUnavailableError(RetryableError):
    """The provider returned a server-side error (HTTP 5xx)."""


class AuthenticationError(NonRetryableError):
    """The API key is missing, invalid or lacks permission (HTTP 401/403)."""


class InvalidRequestError(NonRetryableError):
    """The provider rejected the request as malformed (HTTP 400)."""


class QuotaExhaustedError(NonRetryableError):
    """The account's usage allowance is used up.

    Unlike ``RateLimitError``, waiting a few seconds
    does not help; the quota resets on the provider's
    schedule (e.g. daily), so retrying is wasted work.
    """


class StructuredOutputError(LLMError):
    """The model's reply could not be parsed into the requested schema."""

    def __init__(self, message: str, raw_text: str) -> None:
        """Keep the raw reply so the failure can be debugged later.

        Args:
            message: Human-readable description of the failure.
            raw_text: The model's unparsed reply.
        """
        super().__init__(message)
        self.raw_text = raw_text


class GuardrailError(NonRetryableError):
    """A guardrail rejected the request or the reply.

    Guards are deterministic, so resending the same
    request would be rejected again; hence not retried.
    """

    def __init__(self, guard: str, reason: str) -> None:
        """Record which guard fired and why.

        Args:
            guard: Name of the guard, e.g. ``"max_input_chars"``.
            reason: Short, fixed description of the rule that
                failed. Never include the offending text: errors
                are logged and traced, and that text may be PII
                (Personally Identifiable Information) or a secret.
        """
        super().__init__(f"{guard}: {reason}")
        self.guard = guard
        self.reason = reason
