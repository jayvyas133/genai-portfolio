"""Tests for answer generation with a scripted fake model (no network)."""

import json

import pytest
from aicore.client import LLMClient
from aicore.config import LLMSettings
from aicore.guardrails import SecretLeakGuard
from aicore.providers.fakeai import FakeProvider
from documind.generation import (
    SYSTEM_PROMPT,
    build_request,
    format_sources,
    generate_answer,
    parse_citations,
)
from documind.store import RetrievedChunk

CHUNKS = [
    RetrievedChunk(
        chunk_id=11, document_title="CUDA", section="History", text="Released in 2007.", score=0.9
    ),
    RetrievedChunk(
        chunk_id=22, document_title="Nvidia", section="Products", text="Makes GPUs.", score=0.8
    ),
]


def reply(answer: str, refused: bool = False) -> str:
    """The JSON a well-behaved model returns."""
    return json.dumps({"answer": answer, "refused": refused})


def client_for(fake: FakeProvider) -> LLMClient:
    """An ``LLMClient`` over ``fake`` with no retry delay and no ``.env``."""
    return LLMClient(fake, LLMSettings(_env_file=None), base_delay_s=0)


def test_sources_are_numbered_with_title_and_section_inside_tags() -> None:
    """[1] is the first chunk; the tags wrap all source text."""
    text = format_sources(CHUNKS)
    assert text.startswith("<sources>\n[1] CUDA > History\nReleased in 2007.")
    assert "\n\n[2] Nvidia > Products\nMakes GPUs." in text
    assert text.endswith("</sources>")


def test_request_puts_rules_in_system_and_question_after_sources() -> None:
    """The question comes last, after the untrusted sources, at temperature 0."""
    request = build_request("When?", CHUNKS, max_tokens=100)
    system, user = request.messages
    assert (system.role, system.content) == ("system", SYSTEM_PROMPT)
    assert user.role == "user"
    assert user.content.index("</sources>") < user.content.index("Question: When?")
    assert request.temperature == 0.0
    assert request.max_tokens == 100


def test_parse_citations_keeps_first_mention_order_without_duplicates() -> None:
    """[2] then [1]; the repeated [2] is not listed twice."""
    citations, invalid = parse_citations("A [2]. B [1][2].", CHUNKS)
    assert [(c.number, c.chunk_id, c.document_title) for c in citations] == [
        (2, 22, "Nvidia"),
        (1, 11, "CUDA"),
    ]
    assert invalid == []


@pytest.mark.parametrize("marker", ["[0]", "[3]", "[99]"])
def test_parse_citations_reports_numbers_outside_the_sources(marker: str) -> None:
    """Only 1..len(chunks) are real sources."""
    citations, invalid = parse_citations(f"A {marker}. B [1].", CHUNKS)
    assert [c.number for c in citations] == [1]
    assert invalid == [int(marker.strip("[]"))]


def test_parse_citations_without_markers_is_empty() -> None:
    """Plain text cites nothing."""
    assert parse_citations("No markers here.", CHUNKS) == ([], [])


async def test_generate_answer_returns_text_citations_and_chunks() -> None:
    """A cited answer keeps its text and maps [1] to the first chunk."""
    fake = FakeProvider([reply("  CUDA was released in 2007 [1].  ")])
    answer = await generate_answer("When was CUDA released?", CHUNKS, client_for(fake))
    assert answer.text == answer.draft == "CUDA was released in 2007 [1]."
    assert not answer.refused
    assert not answer.plain_text_reply
    assert [c.chunk_id for c in answer.citations] == [11]
    assert answer.invalid_citations == []
    assert answer.chunks == CHUNKS
    assert "Question: When was CUDA released?" in fake.calls[0].messages[-1].content


async def test_generate_answer_reports_invalid_citation_numbers() -> None:
    """A marker for a source that was never sent is flagged, not shown."""
    fake = FakeProvider([reply("Released in 2007 [1][7].")])
    answer = await generate_answer("When?", CHUNKS, client_for(fake))
    assert [c.number for c in answer.citations] == [1]
    assert answer.invalid_citations == [7]


async def test_model_refusal_drops_any_text_the_model_returned() -> None:
    """refused=true wins: no answer text and no citations are shown."""
    fake = FakeProvider([reply("Maybe 2007 [1].", refused=True)])
    answer = await generate_answer("When?", CHUNKS, client_for(fake))
    assert answer.refused
    assert answer.refusal_reason == "model"
    assert answer.text == ""
    assert answer.citations == []


async def test_no_chunks_refuses_without_calling_the_model() -> None:
    """Nothing retrieved means nothing to cite."""
    fake = FakeProvider()
    answer = await generate_answer("When?", [], client_for(fake))
    assert answer.refusal_reason == "no_sources"
    assert answer.chunks == []
    assert fake.calls == []


async def test_uncited_answer_is_withheld_but_kept_as_draft() -> None:
    """No marker means the answer cannot be traced to a chunk, so it is not shown."""
    fake = FakeProvider([reply("CUDA was released in 2007.")])
    answer = await generate_answer("When?", CHUNKS, client_for(fake))
    assert answer.refusal_reason == "uncited"
    assert answer.text == ""
    assert answer.draft == "CUDA was released in 2007."


async def test_only_invalid_markers_count_as_uncited() -> None:
    """[7] points at no source, so nothing valid is cited."""
    fake = FakeProvider([reply("Released in 2007 [7].")])
    answer = await generate_answer("When?", CHUNKS, client_for(fake))
    assert answer.refusal_reason == "uncited"
    assert answer.invalid_citations == [7]


async def test_plain_text_reply_with_markers_is_used_as_the_answer() -> None:
    """Text instead of JSON is not a failure when it cites valid sources."""
    fake = FakeProvider(["CUDA was released in 2007 [1]."])
    answer = await generate_answer("When?", CHUNKS, client_for(fake))
    assert not answer.refused
    assert answer.plain_text_reply
    assert answer.text == "CUDA was released in 2007 [1]."
    assert [c.number for c in answer.citations] == [1]


async def test_plain_text_reply_without_markers_is_withheld() -> None:
    """The fallback never shows uncited text, such as a prose refusal."""
    fake = FakeProvider(["I cannot find this in the sources."])
    answer = await generate_answer("When?", CHUNKS, client_for(fake))
    assert answer.refusal_reason == "uncited"
    assert answer.plain_text_reply


async def test_reply_blocked_by_an_output_guard_is_a_refusal() -> None:
    """A leaked-looking secret in the reply is never shown, and the guard is named."""
    fake = FakeProvider([reply("The key is AKIAABCDEFGHIJKLMNOP [1].")])
    client = LLMClient(
        fake, LLMSettings(_env_file=None), base_delay_s=0, output_guards=[SecretLeakGuard()]
    )
    answer = await generate_answer("What is the key?", CHUNKS, client)
    assert answer.refusal_reason == "blocked"
    assert answer.blocked_by == "secret_leak"
    assert answer.text == answer.draft == ""
