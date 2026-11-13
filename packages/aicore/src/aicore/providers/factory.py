"""Build the configured ``LLMProvider`` from settings.

This is the one place that maps ``AICORE_PROVIDER`` to an adapter class,
so projects never import a concrete provider directly.
"""

from typing import assert_never

from aicore.config import LLMSettings
from aicore.providers.base import LLMProvider
from aicore.providers.cloudflare import CloudflareProvider
from aicore.providers.fakeai import FakeProvider
from aicore.providers.ollama import OllamaProvider


def build_provider(settings: LLMSettings) -> LLMProvider:
    """Return the adapter named by ``settings.provider``.

    Args:
        settings: Validated settings; ``provider`` picks the adapter.

    Returns:
        A ready-to-use provider. ``OllamaProvider`` holds an HTTP
        connection pool, so call its ``aclose`` when finished.
        A ready-to-use provider. HTTP-based providers
        hold a connection pool, so call their
        ``aclose`` when finished.

    Raises:
        ValueError: If the chosen provider's required
            settings are missing.
    """
    name = settings.provider
    match name:
        case "fake":
            return FakeProvider(model=settings.model)
        case "ollama":
            return OllamaProvider(str(settings.ollama_base_url))
        case "cloudflare":
            return _build_cloudflare(settings)
        case _:
            assert_never(name)


def _build_cloudflare(
    settings: LLMSettings,
) -> CloudflareProvider:
    """Build ``CloudflareProvider``, checking creds first.

    An empty value (``AICORE_API_KEY=`` in ``.env``)
    counts as missing, because pydantic-settings
    reads it as ``""``, not ``None``.

    Raises:
        ValueError: Naming every missing variable.
    """
    token = settings.api_key.get_secret_value() if settings.api_key is not None else ""
    account_id = settings.cloudflare_account_id or ""

    missing: list[str] = []
    if not token:
        missing.append("AICORE_API_KEY")
    if not account_id:
        missing.append("AICORE_CLOUDFLARE_ACCOUNT_ID")
    if missing:
        names = ", ".join(missing)
        msg = f"provider 'cloudflare' needs: {names}"
        raise ValueError(msg)

    return CloudflareProvider(account_id, token)
