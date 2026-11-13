"""Unit tests for ``LLMClient.complete_structured`` and its fence-stripping helper."""

import pytest
from aicore.client import LLMClient, _strip_code_fences
from aicore.config import LLMSettings
from aicore.errors import ProviderUnavailableError, StructuredOutputError
from aicore.models import LLMRequest, Message
from aicore.providers.fakeai import FakeProvider
from pydantic import BaseModel


class Book(BaseModel):
    """Small target schema used by every test in this file."""

    title: str
    year: int


def make_request() -> LLMRequest:
    """Build a minimal one-message request."""
    return LLMRequest(messages=[Message(role="user", content="Name a sci-fi classic.")])


def make_client(provider: FakeProvider) -> LLMClient:
    """Build a client with zero backoff so retry tests run instantly."""
    return LLMClient(provider, LLMSettings(max_retries=3), base_delay_s=0)


async def test_parses_plain_json_into_schema() -> None:
    """A clean JSON reply becomes a validated ``Book`` instance."""
    provider = FakeProvider(script=['{"title": "Dune", "year": 1965}'])

    book = await make_client(provider).complete_structured(make_request(), Book)

    assert book == Book(title="Dune", year=1965)


async def test_parses_json_wrapped_in_code_fence() -> None:
    """A reply wrapped in a ```json fence is still accepted."""
    reply = '```json\n{"title": "Dune", "year": 1965}\n```'
    provider = FakeProvider(script=[reply])

    book = await make_client(provider).complete_structured(make_request(), Book)

    assert book.title == "Dune"


async def test_schema_instruction_is_prepended_as_system_message() -> None:
    """The provider sees the schema first, then the original messages unchanged."""
    provider = FakeProvider(script=['{"title": "Dune", "year": 1965}'])
    request = make_request()

    await make_client(provider).complete_structured(request, Book)

    sent = provider.calls[0].messages
    assert sent[0].role == "system"
    assert '"title"' in sent[0].content and '"year"' in sent[0].content
    assert sent[1:] == request.messages


async def test_wrong_shape_raises_and_keeps_raw_text() -> None:
    """A reply missing a field raises StructuredOutputError and is not retried."""
    raw = '{"title": "Dune"}'
    provider = FakeProvider(script=[raw])

    with pytest.raises(StructuredOutputError) as exc_info:
        await make_client(provider).complete_structured(make_request(), Book)

    assert exc_info.value.raw_text == raw
    assert len(provider.calls) == 1


async def test_non_json_reply_raises_structured_output_error() -> None:
    """Plain prose instead of JSON is reported the same way."""
    provider = FakeProvider(script=["Sorry, I cannot answer that."])

    with pytest.raises(StructuredOutputError):
        await make_client(provider).complete_structured(make_request(), Book)


async def test_transport_errors_are_still_retried() -> None:
    """complete_structured reuses complete(), so transient failures are retried."""
    provider = FakeProvider(
        script=[ProviderUnavailableError("503"), '{"title": "Dune", "year": 1965}']
    )

    book = await make_client(provider).complete_structured(make_request(), Book)

    assert book.year == 1965
    assert len(provider.calls) == 2


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', '{"a": 1}'),
        ('  {"a": 1}  \n', '{"a": 1}'),
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('```\n{"a": 1}\n```', '{"a": 1}'),
    ],
)
def test_strip_code_fences(text: str, expected: str) -> None:
    """Fences and surrounding whitespace are removed; bare JSON is left as is."""
    assert _strip_code_fences(text) == expected
