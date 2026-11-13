"""HTTP helpers shared by aicore's HTTP-based adapters.

Vendors format error bodies differently, so each adapter
extracts its own error text. The mapping from status code
to aicore error lives here, so every adapter classifies
failures the same way.
"""

import httpx

from aicore.errors import (
    AuthenticationError,
    InvalidRequestError,
    LLMError,
    ProviderUnavailableError,
    RateLimitError,
)


def map_status_error(
    response: httpx.Response,
    *,
    vendor: str,
    detail: str,
) -> LLMError:
    """Translate a non-2xx response into an aicore error.

    Args:
        response: The failed HTTP response.
        vendor: Name used in the message, e.g. ``"Ollama"``.
        detail: The vendor's error text, already extracted
            from the response body.

    Returns:
        The error to raise. It is returned, not raised, so
        the ``raise`` stays visible at the call site.
    """
    status = response.status_code
    message = f"{vendor} returned {status}: {detail}"
    if status == 429:
        retry_after = parse_retry_after(response)
        return RateLimitError(message, retry_after)
    if status >= 500:
        return ProviderUnavailableError(message)
    if status in (401, 403):
        return AuthenticationError(message)
    return InvalidRequestError(message)


def parse_retry_after(response: httpx.Response) -> float | None:
    """Read ``Retry-After`` as seconds.

    Returns:
        The wait in seconds, or ``None`` if the header is
        absent or in HTTP-date form.
    """
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        # HTTP-date form: LLMClient uses its own backoff.
        return None
