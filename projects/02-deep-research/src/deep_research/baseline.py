"""Single-shot baseline: one search for the question, then one LLM call.

This is the cheapest research system that could work, and the bar the
agent graph has to clear. If the agents cannot beat it on the eval,
their extra calls are not worth paying for.
"""

from aicore.client import LLMClient

from deep_research.report import Report
from deep_research.search import SearchProvider
from deep_research.writer import build_writer_request


class SingleShotResearcher:
    """``Researcher``: search once with the raw question, write once."""

    name = "single-shot"

    def __init__(self, search: SearchProvider, client: LLMClient, k: int) -> None:
        """Search with ``search`` (``k`` passages) and write with ``client``."""
        self._search = search
        self._client = client
        self._k = k

    async def __call__(self, question: str) -> Report:
        """Research ``question``. With no search hits the model is not called.

        Raises:
            LLMError: If the model call fails after retries.
        """
        hits = await self._search.search(question, self._k)
        passages = [h.passage for h in hits]
        if not passages:
            return Report(question=question, text="No sources were found for this question.")
        response = await self._client.complete(build_writer_request(question, passages))
        return Report(question=question, text=response.text.strip(), sources=passages)
