"""Provider-agnostic request and response models for aicore calls.

Covers chat completion, embedding and reranking. Every provider adapter
translates to and from these types, so project code never depends on a vendor
SDK's objects. Swapping providers therefore never changes calling code.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]

InputType = Literal["query", "document"]
"""Which side of a retrieval pair a text is on (embedding models are asymmetric)."""

NonEmptyStr = Annotated[str, Field(min_length=1)]


class Message(BaseModel):
    """A single chat message with a role and text content."""

    role: Role
    content: str


class LLMRequest(BaseModel):
    """Everything needed to make one LLM call.

    ``model`` is optional; when ``None`` the client falls back to
    ``LLMSettings.model``, so callers only override it when they need to.
    """

    messages: list[Message] = Field(min_length=1)
    model: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)


class Usage(BaseModel):
    """Usage reported by the provider for one call.

    ``neurons`` is Cloudflare's billing unit.
    It stays ``None`` for providers that bill
    by tokens only (or not at all).
    """

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    neurons: float | None = Field(default=None, ge=0)

    @property
    def total_tokens(self) -> int:
        """Sum of input and output tokens."""
        return self.input_tokens + self.output_tokens


class LLMResponse(BaseModel):
    """The normalized result of one LLM call, whatever the provider."""

    text: str
    model: str
    usage: Usage


class EmbedRequest(BaseModel):
    """A batch of texts to turn into vectors in one provider call.

    ``input_type`` tells the adapter whether these are search queries or
    documents, so it can apply the model-specific prefix (if any). ``model``
    falls back to the configured embedding model when ``None``.
    """

    texts: list[NonEmptyStr] = Field(min_length=1)
    input_type: InputType = "document"
    model: str | None = None


class EmbedResponse(BaseModel):
    """Vectors for an ``EmbedRequest``, in the same order as its ``texts``.

    Embedding calls have no generated output, so ``usage.output_tokens`` is 0.
    """

    vectors: list[list[float]]
    model: str
    usage: Usage

    @property
    def dimensions(self) -> int:
        """Length of each vector (0 if the response holds no vectors)."""
        return len(self.vectors[0]) if self.vectors else 0


class RerankRequest(BaseModel):
    """Score candidate documents by their relevance to one query.

    ``top_n`` limits how many results come back; ``None`` returns all of them.
    """

    query: NonEmptyStr
    documents: list[NonEmptyStr] = Field(min_length=1)
    top_n: int | None = Field(default=None, gt=0)
    model: str | None = None


class RerankResult(BaseModel):
    """Relevance of one document, identified by its index in the request."""

    index: int = Field(ge=0)
    score: float


class RerankResponse(BaseModel):
    """Rerank results sorted by ``score``, highest first."""

    results: list[RerankResult]
    model: str
    usage: Usage
