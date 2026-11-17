"""Configuration for DocuMind.

Values come from environment variables prefixed with ``DOCUMIND_`` (or
a local ``.env`` file). Model choice and API keys are not here: they
belong to aicore (``AICORE_*``), which makes every model call.
"""

from pathlib import Path
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# projects/01-documind/ (this file is src/documind/config.py).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DocumindSettings(BaseSettings):
    """Settings for ingestion and retrieval.

    Example: ``DOCUMIND_CHUNK_WORDS=200`` sets ``chunk_words`` to 200.

    ``database_url`` points at a dedicated ``documind`` database in the
    shared pgvector container. ``embedding_dimensions`` must match the
    embedding model (768 for ``@cf/baai/bge-base-en-v1.5``); the schema
    is created with it and ingestion checks every vector against it.
    """

    model_config = SettingsConfigDict(
        env_prefix="DOCUMIND_",
        env_file=".env",
        extra="ignore",
    )

    database_url: str = "postgresql://postgres:postgres@localhost:5432/documind"
    data_dir: Path = _PROJECT_ROOT / "data"
    # 200 words keeps the densest chunk (~1.9 tokens/word) under BGE's 512-token limit.
    chunk_words: int = Field(default=200, gt=0)
    chunk_overlap_words: int = Field(default=40, ge=0)
    embed_batch_size: int = Field(default=50, gt=0, le=100)
    embedding_dimensions: int = Field(default=768, gt=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> Self:
        """Reject an overlap that would stop the chunker from advancing."""
        if self.chunk_overlap_words >= self.chunk_words:
            msg = "chunk_overlap_words must be smaller than chunk_words"
            raise ValueError(msg)
        return self
