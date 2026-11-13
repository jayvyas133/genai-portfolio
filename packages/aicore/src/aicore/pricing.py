"""Turn reported usage into a cost in US dollars.

Providers only report usage (tokens, neurons);
aicore applies prices here, so every project
computes cost the same way.
"""

from aicore.models import Usage

# Workers AI list price: USD 0.011 per 1,000
# neurons. The free plan (10,000 neurons/day) is
# not charged; this is what the call would cost
# on the paid plan.
CLOUDFLARE_USD_PER_NEURON = 0.011 / 1000

# Providers that run on our own machine.
_LOCAL_PROVIDERS = frozenset({"fake", "ollama"})


def cost_usd(provider: str, usage: Usage) -> float | None:
    """Return the cost of one call, or ``None``.

    Args:
        provider: Provider name, as in
            ``AICORE_PROVIDER``.
        usage: Usage reported for the call.

    Returns:
        ``0.0`` for local providers, a dollar
        amount when the price is known, and
        ``None`` when it is not. ``None`` is never
        replaced by 0, so an unknown cost is never
        reported as free.
    """
    if provider in _LOCAL_PROVIDERS:
        return 0.0
    if provider == "cloudflare":
        if usage.neurons is None:
            return None
        return usage.neurons * CLOUDFLARE_USD_PER_NEURON
    return None
