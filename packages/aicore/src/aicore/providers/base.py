"""The contract every LLM provider adapter must satisfy.

``LLMClient`` depends only on this protocol, never on a vendor SDK. Adding a
new backend therefore means writing one adapter class; no calling code changes.
"""

from typing import Protocol

from aicore.models import LLMRequest, LLMResponse


class LLMProvider(Protocol):
    """Structural interface for an LLM backend.

    Any class with a matching ``complete`` method satisfies this protocol;
    inheriting from it is not required. mypy verifies the match statically.
    """

    name: str
    """Short provider ID, as in ``AICORE_PROVIDER``."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send one request to the backend and return a normalized response.

        Implementations must translate vendor errors into aicore's error types
        so the client can decide whether a failure is worth retrying.
        """
        ...
