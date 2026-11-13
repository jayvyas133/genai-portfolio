"""Provider-agnostic request and response models for the LLM client.

Every provider adapter translates to and from these types, so project code
never depends on a vendor SDK's objects. Swapping providers therefore never
changes calling code.
"""

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


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
    """Token counts reported by the provider for one call"""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)

    @property
    def total_tokens(self) -> int:
        """Sum of input and output tokens."""
        return self.input_tokens + self.output_tokens


class LLMResponse(BaseModel):
    """The normalized result of one LLM call, whatever the provider."""

    text: str
    model: str
    usage: Usage
