"""Configuration for aicore's LLM client.

All values are read from environment variables prefixed with ``AICORE_``
(or a local ``.env`` file), so the same code runs unchanged in dev, CI
and production. Values are validated at startup to fail fast on bad config.
"""

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """Settings for the LLM client.

    Example: ``AICORE_TIMEOUT_S=10`` sets ``timeout_s`` to 10 seconds.
    """

    model_config = SettingsConfigDict(
        env_prefix="AICORE_",
        env_file=".env",
        extra="ignore",
    )

    provider: Literal["fake"] = "fake"
    model: str = "fake-model"
    api_key: SecretStr | None = None
    timeout_s: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
