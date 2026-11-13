"""Unit tests for ``cost_usd``."""

import pytest
from aicore.models import Usage
from aicore.pricing import CLOUDFLARE_USD_PER_NEURON, cost_usd


def usage(neurons: float | None = None) -> Usage:
    """Return a small usage with optional neurons."""
    return Usage(input_tokens=10, output_tokens=5, neurons=neurons)


@pytest.mark.parametrize("provider", ["fake", "ollama"])
def test_local_providers_cost_zero(provider: str) -> None:
    """Local providers have no per-call bill."""
    assert cost_usd(provider, usage()) == 0.0


def test_cloudflare_priced_by_neurons() -> None:
    """Cost is neurons times the list price."""
    cost = cost_usd("cloudflare", usage(neurons=1000))
    assert cost == pytest.approx(1000 * CLOUDFLARE_USD_PER_NEURON)


def test_cloudflare_without_neurons_is_unknown() -> None:
    """Missing neurons gives ``None``, never 0."""
    assert cost_usd("cloudflare", usage()) is None


def test_unknown_provider_is_unknown() -> None:
    """A provider with no price gives ``None``."""
    assert cost_usd("gemini", usage()) is None
