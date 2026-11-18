"""Run DocuMind's async pipeline from synchronous code (the CLI and Flask views)."""

import asyncio
import sys
from collections.abc import Callable, Coroutine
from typing import Any


def _loop_factory() -> Callable[[], asyncio.AbstractEventLoop] | None:
    """Return a selector loop on Windows; psycopg async cannot use Proactor."""
    return asyncio.SelectorEventLoop if sys.platform == "win32" else None


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` to completion on a new event loop and return its result.

    Safe to call from any thread that has no running loop, such as a
    Flask or gunicorn worker thread.
    """
    return asyncio.run(coro, loop_factory=_loop_factory())
