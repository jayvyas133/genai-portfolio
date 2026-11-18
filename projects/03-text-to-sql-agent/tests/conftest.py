"""Shared pytest setup for the Text-to-SQL agent tests."""

import asyncio
import sys
from collections.abc import Callable, Mapping

import pytest


def pytest_asyncio_loop_factories(
    config: pytest.Config, item: pytest.Item
) -> Mapping[str, Callable[[], asyncio.AbstractEventLoop]] | None:
    """Run async tests on a selector event loop on Windows.

    psycopg's async mode cannot run on Windows' default Proactor loop;
    Linux and macOS already default to a selector loop (``None`` keeps it).
    """
    if sys.platform == "win32":
        return {"selector": asyncio.SelectorEventLoop}
    return None
