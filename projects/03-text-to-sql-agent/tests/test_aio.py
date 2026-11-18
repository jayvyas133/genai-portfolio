"""Tests for the background event-loop thread used by the Flask API."""

import asyncio
import threading
from collections.abc import Iterator

import pytest
from text_to_sql_agent.aio import LoopThread


@pytest.fixture
def loop() -> Iterator[LoopThread]:
    thread = LoopThread()
    yield thread
    thread.stop()


def test_runs_coroutines_on_one_long_lived_loop(loop: LoopThread) -> None:
    async def which_loop() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    assert loop.run(which_loop()) is loop.run(which_loop()) is loop.loop


def test_exceptions_reach_the_caller(loop: LoopThread) -> None:
    async def boom() -> None:
        raise ValueError("bad")

    with pytest.raises(ValueError, match="bad"):
        loop.run(boom())


def test_timeout_raises_and_cancels_the_coroutine(loop: LoopThread) -> None:
    cancelled = threading.Event()

    async def hang() -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with pytest.raises(TimeoutError):
        loop.run(hang(), timeout_s=0.1)
    assert cancelled.wait(2)


def test_many_threads_can_submit_at_once(loop: LoopThread) -> None:
    results: list[int] = []

    async def square(n: int) -> int:
        await asyncio.sleep(0.01)
        return n * n

    def worker(n: int) -> None:
        results.append(loop.run(square(n)))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == [n * n for n in range(8)]
