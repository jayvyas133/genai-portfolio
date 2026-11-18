"""Run async code from synchronous code: the CLI, and Flask views.

Two tools, for two situations:

- ``run``: a fresh event loop for one coroutine. Right for the CLI, which
  does one thing and exits.
- ``LoopThread``: one long-lived event loop in a background thread. The
  Flask API needs this, because connection pools, the model client and
  the agent graph are bound to the loop that created them. Creating them
  once on a shared loop lets every request reuse them; ``asyncio.run``
  per request (DocuMind's approach) would force a new connection and
  client per request.
"""

import asyncio
import sys
import threading
from collections.abc import Callable, Coroutine
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any


def _loop_factory() -> Callable[[], asyncio.AbstractEventLoop] | None:
    """Return a selector loop on Windows; psycopg async cannot use Proactor."""
    return asyncio.SelectorEventLoop if sys.platform == "win32" else None


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` to completion on a new event loop and return its result."""
    return asyncio.run(coro, loop_factory=_loop_factory())


class LoopThread:
    """An event loop running forever in a daemon thread, fed from other threads.

    Thread-safe: any number of request threads may call ``run`` at once;
    their coroutines interleave on the one loop.
    """

    def __init__(self) -> None:
        """Start the loop and its thread."""
        factory = _loop_factory() or asyncio.new_event_loop
        self.loop = factory()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True, name="aio")
        self._thread.start()

    def run[T](self, coro: Coroutine[Any, Any, T], timeout_s: float | None = None) -> T:
        """Run ``coro`` on the loop and wait for its result.

        Raises:
            TimeoutError: After ``timeout_s``; the coroutine is cancelled, so
                its work (and any lock it holds) is released.
            Exception: Whatever ``coro`` raised.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return future.result(timeout_s)
        except FutureTimeout as err:
            future.cancel()
            raise TimeoutError(f"no result within {timeout_s} s") from err

    def stop(self) -> None:
        """Stop the loop and wait for its thread to finish."""
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)
        self.loop.close()
