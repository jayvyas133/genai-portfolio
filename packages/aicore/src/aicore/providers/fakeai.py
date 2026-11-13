"""A scripted, offline provider for tests and local development.

``FakeProvider`` needs no network or API key. It replays a scripted sequence
of replies and errors, so tests can exercise the client's retry logic
deterministically.
"""

from collections.abc import Sequence

from aicore.models import LLMRequest, LLMResponse, Usage

Outcome = str | Exception
"""One scripted result: a ``str`` is returned as reply text, an ``Exception`` is raised."""


class FakeProvider:
    """In-memory ``LLMProvider`` that replays scripted outcomes.

    Each call to ``complete`` consumes the next outcome from the script. Once
    the script is exhausted, every call returns ``default_reply``.

    Attributes:
        calls: Every request received, in order, so tests can assert on how
            many times (and with what) the provider was called.
    """

    name = "fake"

    def __init__(
        self,
        script: Sequence[Outcome] = (),
        default_reply: str = "fake reply",
        model: str = "fake-model",
    ) -> None:
        """Create a provider that will replay ``script`` in order.

        Args:
            script: Outcomes to return or raise, one per call.
            default_reply: Reply text used after the script runs out.
            model: Model name reported when the request does not set one.
        """
        self._script = list(script)  # copy, so the caller's list is never mutated
        self._default_reply = default_reply
        self._model = model
        self.calls: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Record the request, then return or raise the next scripted outcome.

        Token counts are approximated by word counts. That is good enough for
        tests and keeps this class free of tokenizer dependencies.

        Raises:
            Exception: Whatever exception is next in the script.
        """
        self.calls.append(request)
        outcome = self._script.pop(0) if self._script else self._default_reply
        if isinstance(outcome, Exception):
            raise outcome
        input_tokens = sum(len(m.content.split()) for m in request.messages)
        return LLMResponse(
            text=outcome,
            model=request.model or self._model,
            usage=Usage(input_tokens=input_tokens, output_tokens=len(outcome.split())),
        )
