"""Tests for the corpus snapshot: cleaning, fetching and loading."""

import json
from pathlib import Path

import httpx
import pytest
from deep_research.corpus import (
    Document,
    fetch_document,
    load_documents,
    slugify,
    strip_trailing_sections,
)


def test_slugify_is_file_name_safe() -> None:
    assert slugify("Chang'e 4") == "change-4"
    assert slugify("Orion (spacecraft)") == "orion-spacecraft"


def test_strip_trailing_sections_drops_reference_sections_and_their_subsections() -> None:
    text = (
        "Intro.\n\n== History ==\nFacts.\n\n== See also ==\nLinks.\n"
        "=== More links ===\nMore.\n\n== Legacy ==\nKept.\n\n== References ==\nRefs."
    )
    cleaned = strip_trailing_sections(text)
    assert "Facts." in cleaned
    assert "== Legacy ==\nKept." in cleaned
    for dropped in ("See also", "Links.", "More links", "More.", "References", "Refs."):
        assert dropped not in cleaned


def _page(title: str = "Luna 25") -> dict[str, object]:
    """A minimal Wikipedia API reply for one page."""
    return {
        "query": {
            "pages": [
                {
                    "title": title,
                    "fullurl": f"https://en.wikipedia.org/wiki/{title}",
                    "extract": "Text.\n\n== Notes ==\nDropped.",
                }
            ]
        }
    }


async def test_fetch_document_retries_after_429_then_parses_the_page() -> None:
    replies = [
        httpx.Response(429, headers={"Retry-After": "0"}),
        httpx.Response(200, json=_page()),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return replies.pop(0)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        doc = await fetch_document(client, "Luna 25")
    assert doc == Document(
        title="Luna 25", url="https://en.wikipedia.org/wiki/Luna 25", text="Text."
    )


async def test_fetch_document_raises_for_a_missing_page() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"query": {"pages": [{"title": "X", "missing": True}]}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(LookupError):
            await fetch_document(client, "X")


def test_load_documents_sorts_by_file_name(tmp_path: Path) -> None:
    for name in ("b", "a"):
        doc = {"title": name.upper(), "url": "u", "text": "t"}
        (tmp_path / f"{name}.json").write_text(json.dumps(doc), encoding="utf-8")
    assert [d.title for d in load_documents(tmp_path)] == ["A", "B"]


def test_load_documents_rejects_missing_or_empty_folders(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope")
    with pytest.raises(ValueError, match="No documents"):
        load_documents(tmp_path)
