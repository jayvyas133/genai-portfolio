"""Tests for the section-aware chunker. Pure functions, so no fixtures needed."""

import pytest
from documind.chunking import (
    INTRO_SECTION,
    Chunk,
    Section,
    chunk_document,
    split_sections,
    window_words,
)
from documind.config import DocumindSettings
from documind.corpus import Document, load_documents
from pydantic import ValidationError

ARTICLE = """CUDA is a parallel computing platform.

== History ==
CUDA was released in 2007.

=== Early years ===
The first version supported G80 GPUs.

== Architecture ==
=== Threads ===
Threads are grouped into warps.

== See also ==
OpenCL

=== Related ===
ROCm

== References ==
Some citation."""


def words(n: int, start: int = 0) -> str:
    """Return ``n`` distinct words ``w<start> ... w<start+n-1>``."""
    return " ".join(f"w{i}" for i in range(start, start + n))


class TestSplitSections:
    """Heading paths, empty sections and boilerplate removal."""

    def test_paths_follow_heading_levels(self) -> None:
        """Lead text is ``Introduction``; subsections nest under their parent."""
        assert split_sections(ARTICLE) == [
            Section(INTRO_SECTION, "CUDA is a parallel computing platform."),
            Section("History", "CUDA was released in 2007."),
            Section("History > Early years", "The first version supported G80 GPUs."),
            Section("Architecture > Threads", "Threads are grouped into warps."),
        ]

    def test_shallower_heading_resets_deeper_path(self) -> None:
        """``== B ==`` after ``=== A1 ===`` starts a fresh path, not ``A > B``."""
        text = "== A ==\n=== A1 ===\nx\n== B ==\ny"

        assert [s.path for s in split_sections(text)] == ["A > A1", "B"]

    def test_skipped_level_does_not_crash(self) -> None:
        """A ``====`` heading directly under ``==`` still gets a sensible path."""
        text = "== A ==\n==== Deep ====\nx"

        assert split_sections(text) == [Section("A > Deep", "x")]

    def test_boilerplate_and_its_subsections_dropped(self) -> None:
        """See also / References (and their children) never become sections."""
        paths = [s.path for s in split_sections(ARTICLE)]

        assert not any("See also" in p or "Related" in p or "References" in p for p in paths)

    def test_text_without_headings_is_one_intro_section(self) -> None:
        """A document with no headings is all lead text."""
        assert split_sections("Just text.") == [Section(INTRO_SECTION, "Just text.")]

    @pytest.mark.parametrize("line", ["x == y", "== Open only", "=Single=", "a = b == c"])
    def test_equals_signs_in_body_are_not_headings(self, line: str) -> None:
        """Only a whole line wrapped in matching ``==`` markers is a heading."""
        text = f"lead\n{line}\nmore"

        assert split_sections(text) == [Section(INTRO_SECTION, text)]

    def test_empty_text(self) -> None:
        """Nothing in, nothing out."""
        assert split_sections("") == []


class TestWindowWords:
    """Sliding windows: sizes, overlap and the stop condition."""

    def test_short_text_is_one_window(self) -> None:
        """Text up to ``size`` words is returned whole."""
        assert window_words(words(5), size=5, overlap=2) == [words(5)]

    def test_windows_overlap_and_cover_everything(self) -> None:
        """12 words, size 5, overlap 2 -> starts 0, 3, 6, 9."""
        windows = window_words(words(12), size=5, overlap=2)

        assert windows == [words(5, 0), words(5, 3), words(5, 6), words(3, 9)]

    def test_no_window_of_pure_overlap(self) -> None:
        """When a window reaches the end, no extra all-overlap window follows."""
        windows = window_words(words(8), size=5, overlap=2)

        assert windows == [words(5, 0), words(5, 3)]

    def test_zero_overlap(self) -> None:
        """Overlap 0 gives back-to-back windows."""
        assert window_words(words(4), size=2, overlap=0) == [words(2, 0), words(2, 2)]

    def test_whitespace_is_normalised(self) -> None:
        """Newlines and runs of spaces become single spaces."""
        assert window_words("a\n\nb   c", size=10, overlap=0) == ["a b c"]

    def test_empty_text_gives_no_windows(self) -> None:
        """Whitespace-only text produces nothing."""
        assert window_words("  \n ", size=5, overlap=1) == []

    @pytest.mark.parametrize(("size", "overlap"), [(0, 0), (5, 5), (5, 6), (5, -1)])
    def test_invalid_sizes_rejected(self, size: int, overlap: int) -> None:
        """Bad settings fail loudly instead of looping forever."""
        with pytest.raises(ValueError, match="overlap"):
            window_words("a b c", size=size, overlap=overlap)


class TestChunk:
    """``embed_text`` header and ``content_hash`` identity."""

    def test_embed_text_has_title_and_section_header(self) -> None:
        """The header gives the vector context the raw text lacks."""
        chunk = Chunk(document_title="CUDA", ordinal=0, section="History", text="It launched.")

        assert chunk.embed_text == "CUDA > History\nIt launched."

    def test_hash_is_stable_and_ignores_ordinal(self) -> None:
        """Same content -> same hash, even if the chunk moved position."""
        a = Chunk(document_title="T", ordinal=0, section="S", text="x")
        b = Chunk(document_title="T", ordinal=7, section="S", text="x")

        assert a.content_hash == b.content_hash
        assert len(a.content_hash) == 64

    @pytest.mark.parametrize("change", [{"document_title": "U"}, {"section": "R"}, {"text": "y"}])
    def test_hash_changes_with_embedded_content(self, change: dict[str, str]) -> None:
        """Title, section or text changes mean the vector must be recomputed."""
        base = Chunk(document_title="T", ordinal=0, section="S", text="x")

        assert base.model_copy(update=change).content_hash != base.content_hash

    def test_empty_text_rejected(self) -> None:
        """A chunk must contain something to embed."""
        with pytest.raises(ValidationError):
            Chunk(document_title="T", ordinal=0, section="S", text="")


class TestChunkDocument:
    """End to end over one document and over the real corpus."""

    def test_ordinals_sections_and_title(self) -> None:
        """Chunks are numbered in reading order and keep their section path."""
        doc = Document(title="CUDA", url="u", text=ARTICLE)

        chunks = chunk_document(doc, size=300, overlap=50)

        assert [(c.ordinal, c.section) for c in chunks] == [
            (0, INTRO_SECTION),
            (1, "History"),
            (2, "History > Early years"),
            (3, "Architecture > Threads"),
        ]
        assert {c.document_title for c in chunks} == {"CUDA"}

    def test_chunks_never_cross_sections(self) -> None:
        """A long section is windowed on its own; its words never leak into the next."""
        doc = Document(title="T", url="u", text=f"== A ==\n{words(7)}\n== B ==\nbee")

        chunks = chunk_document(doc, size=5, overlap=1)

        assert [(c.section, c.text) for c in chunks] == [
            ("A", words(5, 0)),
            ("A", words(3, 4)),
            ("B", "bee"),
        ]

    def test_real_corpus(self) -> None:
        """The committed corpus chunks within limits and without duplicates."""
        s = DocumindSettings(_env_file=None)
        chunks = [
            c
            for d in load_documents(s.data_dir / "raw")
            for c in chunk_document(d, s.chunk_words, s.chunk_overlap_words)
        ]

        assert 500 < len(chunks) < 1000
        assert max(len(c.text.split()) for c in chunks) <= s.chunk_words
        assert len({c.content_hash for c in chunks}) == len(chunks)
        assert not any(c.section.startswith("References") for c in chunks)
