"""Configuration for Deep Research.

Values come from environment variables prefixed with ``DEEP_RESEARCH_``
(or a local ``.env`` file). Model choice and API keys are not here: they
belong to aicore (``AICORE_*``), which makes every model call.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# projects/02-deep-research/ (this file is src/deep_research/config.py).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DeepResearchSettings(BaseSettings):
    """Settings for search and evals.

    Example: ``DEEP_RESEARCH_SEARCH_K=8`` sets ``search_k`` to 8.
    """

    model_config = SettingsConfigDict(
        env_prefix="DEEP_RESEARCH_",
        env_file=".env",
        extra="ignore",
    )

    # Frozen snapshot of source documents, one JSON file each.
    corpus_dir: Path = _PROJECT_ROOT / "data" / "corpus"
    # Golden dataset (golden.jsonl) and saved eval results (results/).
    evals_dir: Path = _PROJECT_ROOT / "evals"
    # Passages returned per search query.
    search_k: int = Field(default=5, gt=0, le=50)
    # Target passage length. ~120 words is one or two paragraphs: long
    # enough to hold a fact with its context, short enough that five of
    # them fit a small local model's context window with room to spare.
    passage_words: int = Field(default=120, gt=20)
