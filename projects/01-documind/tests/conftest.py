"""Shared pytest setup for DocuMind tests."""

import asyncio
import sys
from collections.abc import Callable, Mapping

import pytest

if sys.platform == "win32":
    # Defined only on Windows: pytest-asyncio rejects a hook that returns None,
    # and Linux/macOS already default to a selector loop.

    def pytest_asyncio_loop_factories(
        config: pytest.Config, item: pytest.Item
    ) -> Mapping[str, Callable[[], asyncio.AbstractEventLoop]]:
        """Run async tests on a selector event loop on Windows.

        psycopg's async mode cannot run on Windows' default Proactor loop.
        """
        return {"selector": asyncio.SelectorEventLoop}
