"""Tests for report citation parsing."""

from deep_research.report import (
    Claim,
    Report,
    invalid_markers,
    markers,
    split_claims,
    strip_markers,
)
from deep_research.search import Passage


def test_markers_lists_every_number_in_order() -> None:
    assert markers("A [2]. B [1][3]. C [2].") == [2, 1, 3, 2]


def test_split_claims_keeps_markers_with_the_sentence_they_follow() -> None:
    text = "It launched in 2019 [1]. It crashed [2][1]. No source here.\n\nNew paragraph [3]."
    assert split_claims(text) == [
        Claim(sentence="It launched in 2019 [1].", citations=[1]),
        Claim(sentence="It crashed [2][1].", citations=[2, 1]),
        Claim(sentence="No source here.", citations=[]),
        Claim(sentence="New paragraph [3].", citations=[3]),
    ]


def test_split_claims_handles_markers_after_the_full_stop_and_bullets() -> None:
    text = "Landed on Mars. [1] Then drove.\n- Bullet one [2]."
    sentences = [c.sentence for c in split_claims(text)]
    assert sentences == ["Landed on Mars. [1]", "Then drove.", "- Bullet one [2]."]


def test_invalid_markers_are_those_outside_the_sources() -> None:
    source = Passage(id="a#0", title="A", section="S", url="u", text="t")
    report = Report(question="q", text="X [1]. Y [2]. Z [0].", sources=[source])
    assert invalid_markers(report) == [2, 0]


def test_strip_markers_removes_markers_and_extra_spaces() -> None:
    assert strip_markers("It landed [1] in 2023 [2][3].") == "It landed in 2023."
