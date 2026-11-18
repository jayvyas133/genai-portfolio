"""Tests for passage splitting and BM25 search."""

from deep_research.corpus import Document
from deep_research.search import (
    BM25Index,
    CorpusSearch,
    FakeSearch,
    Passage,
    SearchHit,
    split_passages,
    tokenize,
)


def _passage(pid: str, text: str, title: str = "T") -> Passage:
    return Passage(id=pid, title=title, section="S", url="u", text=text)


def test_tokenize_lowercases_and_drops_stopwords() -> None:
    assert tokenize("When did the PSLV first fly?") == ["pslv", "first", "fly"]


def test_split_passages_packs_paragraphs_and_never_crosses_a_heading() -> None:
    text = "one two three\n\nfour five\n\n== Launch ==\nsix seven"
    doc = Document(title="Luna 25", url="u", text=text)
    passages = split_passages(doc, target_words=5)
    assert [(p.id, p.section, p.text) for p in passages] == [
        ("luna-25#0", "Introduction", "one two three four five"),
        ("luna-25#1", "Launch", "six seven"),
    ]


def test_split_passages_starts_a_new_passage_when_the_next_paragraph_overflows() -> None:
    doc = Document(title="T", url="u", text="a b c d\n\ne f g")
    assert [p.text for p in split_passages(doc, target_words=5)] == ["a b c d", "e f g"]


def test_split_passages_cuts_a_very_long_paragraph() -> None:
    doc = Document(title="T", url="u", text=" ".join(f"w{i}" for i in range(25)))
    assert [len(p.text.split()) for p in split_passages(doc, target_words=5)] == [10, 10, 5]


def test_bm25_ranks_the_matching_passage_first() -> None:
    index = BM25Index(
        [
            _passage("a", "The rover landed in Gale crater on Mars."),
            _passage("b", "The telescope orbits the Sun near L2."),
            _passage("c", "The rocket launched from Florida."),
        ]
    )
    hits = index.top("Gale crater rover", 2)
    assert [h.passage.id for h in hits] == ["a"]  # passages with no query term are left out
    assert hits[0].score > 0


def test_bm25_rare_terms_outweigh_common_ones() -> None:
    index = BM25Index(
        [
            _passage("common", "rocket rocket launch"),
            _passage("rare", "rocket cryogenic"),
            _passage("x", "rocket"),
        ]
    )
    assert index.top("rocket cryogenic", 1)[0].passage.id == "rare"


def test_bm25_indexes_the_title() -> None:
    index = BM25Index(
        [_passage("a", "It launched in 2023.", title="Chandrayaan-3"), _passage("b", "x")]
    )
    assert index.top("chandrayaan", 5)[0].passage.id == "a"


def test_bm25_returns_nothing_for_unknown_or_stopword_queries() -> None:
    index = BM25Index([_passage("a", "rocket")])
    assert index.top("the of and", 5) == []
    assert index.top("zebra", 5) == []
    assert index.top("rocket", 0) == []


async def test_corpus_search_finds_passages_across_documents() -> None:
    docs = [
        Document(title="Luna 25", url="u", text="Luna 25 crashed on the Moon."),
        Document(title="Chandrayaan-3", url="u", text="Chandrayaan-3 landed near the south pole."),
    ]
    hits = await CorpusSearch(docs, passage_words=50).search("south pole landing", 5)
    assert hits[0].passage.title == "Chandrayaan-3"


async def test_fake_search_records_queries_and_truncates() -> None:
    hits = [SearchHit(passage=_passage(str(i), "t"), score=1.0) for i in range(3)]
    fake = FakeSearch(hits)
    assert len(await fake.search("q", 2)) == 2
    assert fake.queries == ["q"]
