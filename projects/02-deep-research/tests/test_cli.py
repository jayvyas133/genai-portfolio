"""Tests for the command line."""

import pytest
from deep_research.cli import build_parser, main


def test_eval_options_parse() -> None:
    args = build_parser().parse_args(
        ["eval", "--split", "dev", "--limit", "3", "--max-claims", "0"]
    )
    assert (args.split, args.limit, args.max_claims, args.judge_model) == ("dev", 3, 0, None)


def test_search_prints_ranked_passages_from_the_corpus(capsys: pytest.CaptureFixture[str]) -> None:
    main(["search", "Luna 25 crash", "-k", "2"])
    out = capsys.readouterr().out
    assert "luna-25#" in out
