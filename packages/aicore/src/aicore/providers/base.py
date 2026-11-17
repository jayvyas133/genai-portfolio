"""The contracts that provider adapters satisfy.

``LLMClient`` depends only on these protocols, never on a vendor SDK. Adding a
new backend therefore means writing one adapter class; no calling code changes.

Capabilities are split into separate protocols (chat completion, embedding,
reranking) because not every backend offers all three. Ollama, for example,
has no rerank endpoint. An adapter implements only the protocols it supports,
and mypy rejects wiring it where an unsupported capability is required.
"""

from typing import Protocol

from aicore.models import (
    EmbedRequest,
    EmbedResponse,
    LLMRequest,
    LLMResponse,
    RerankRequest,
    RerankResponse,
)


class LLMProvider(Protocol):
    """Structural interface for a chat-completion backend.

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


class EmbeddingProvider(Protocol):
    """Structural interface for a backend that turns texts into vectors."""

    name: str
    """Short provider ID, as in ``AICORE_PROVIDER``."""

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        """Embed every text in ``request.texts`` in one call.

        Vectors must come back in the same order as the input texts. The
        adapter applies any model-specific prefix for ``request.input_type``
        and translates vendor errors into aicore's error types.
        """
        ...


class RerankProvider(Protocol):
    """Structural interface for a backend that scores documents against a query."""

    name: str
    """Short provider ID, as in ``AICORE_PROVIDER``."""

    async def rerank(self, request: RerankRequest) -> RerankResponse:
        """Score ``request.documents`` by relevance to ``request.query``.

        Results must be sorted by score, highest first, truncated to
        ``request.top_n`` when set, and reference documents by their index.
        Vendor errors are translated into aicore's error types.
        """
        ...
