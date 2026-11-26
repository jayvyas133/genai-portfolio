"""Configuration for aicore's LLM client.

All values are read from environment variables prefixed with ``AICORE_``
(or a local ``.env`` file), so the same code runs unchanged in dev, CI
and production. Values are validated at startup to fail fast on bad config.
"""

from typing import Literal

from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
    
ProviderName = Literal["fake", "ollama", "cloudflare"]


class LLMSettings(BaseSettings):
    """Settings for the LLM client.

    Example: ``AICORE_TIMEOUT_S=10`` sets ``timeout_s`` to 10 seconds.
    Cloudflare uses ``api_key`` for its API token and
    ``cloudflare_account_id`` for the account in the URL.
    ``cache_ttl_s`` is how long a cached reply lives
    (default one day).
    """

    model_config = SettingsConfigDict(
        env_prefix="AICORE_",
        env_file=".env",
        extra="ignore",
    )

    provider: ProviderName = "fake"
    model: str = "fake-model"
    embed_model: str = "fake-model"
    rerank_model: str = "fake-model"
    api_key: SecretStr | None = None
    timeout_s: float = Field(default=30.0, gt=0)
    cache_ttl_s: int = Field(default=86_400, gt=0)
    max_retries: int = Field(default=3, ge=0)
    ollama_base_url: HttpUrl = HttpUrl("http://localhost:11434")
    cloudflare_account_id: str | None = None
