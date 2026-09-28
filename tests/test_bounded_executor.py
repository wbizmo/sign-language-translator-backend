import asyncio
import threading

import pytest

from src.api.bounded_executor import BoundedExecutor, ExecutionQueueFull


def test_executor_keeps_blocking_work_off_the_event_loop() -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking_work() -> str:
        started.set()
        release.wait(timeout=2)
        return "done"

    async def scenario() -> None:
        executor = BoundedExecutor(max_concurrency=1, max_queue=1, name="test")
        await executor.start()
        try:
            task = asyncio.create_task(executor.submit(blocking_work))
            assert await asyncio.to_thread(started.wait, 1)

            ticked = False

            async def ticker() -> None:
                nonlocal ticked
                await asyncio.sleep(0)
                ticked = True
                release.set()

            await ticker()
            assert await task == "done"
            assert ticked
        finally:
            release.set()
            await executor.close()

    asyncio.run(scenario())


def test_executor_rejects_when_running_and_waiting_capacity_is_full() -> None:
    started = threading.Event()
    release = threading.Event()

    def first_work() -> str:
        started.set()
        release.wait(timeout=2)
        return "first"

    async def scenario() -> None:
        executor = BoundedExecutor(max_concurrency=1, max_queue=1, name="test")
        await executor.start()
        try:
            first = asyncio.create_task(executor.submit(first_work))
            assert await asyncio.to_thread(started.wait, 1)

            second = asyncio.create_task(executor.submit(lambda: "second"))
            for _ in range(50):
                if executor.queue_size == 1:
                    break
                await asyncio.sleep(0)
            assert executor.queue_size == 1

            with pytest.raises(ExecutionQueueFull):
                await executor.submit(lambda: "third")

            release.set()
            assert await first == "first"
            assert await second == "second"
        finally:
            release.set()
            await executor.close()

    asyncio.run(scenario())


def test_cancelled_queued_work_is_not_executed() -> None:
    started = threading.Event()
    release = threading.Event()
    queued_ran = threading.Event()

    def first_work() -> None:
        started.set()
        release.wait(timeout=2)

    def queued_work() -> None:
        queued_ran.set()

    async def scenario() -> None:
        executor = BoundedExecutor(max_concurrency=1, max_queue=1, name="test")
        await executor.start()
        try:
            first = asyncio.create_task(executor.submit(first_work))
            assert await asyncio.to_thread(started.wait, 1)

            queued = asyncio.create_task(executor.submit(queued_work))
            for _ in range(50):
                if executor.queue_size == 1:
                    break
                await asyncio.sleep(0)
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued

            release.set()
            await first
            await asyncio.sleep(0.01)
            assert not queued_ran.is_set()
        finally:
            release.set()
            await executor.close()

    asyncio.run(scenario())


def test_worker_exception_is_propagated() -> None:
    def fail() -> None:
        raise RuntimeError("boom")

    async def scenario() -> None:
        executor = BoundedExecutor(max_concurrency=1, max_queue=1, name="test")
        await executor.start()
        try:
            with pytest.raises(RuntimeError, match="boom"):
                await executor.submit(fail)
        finally:
            await executor.close()

    asyncio.run(scenario())


def test_invalid_limits_are_rejected() -> None:
    with pytest.raises(ValueError):
        BoundedExecutor(max_concurrency=0, max_queue=1, name="test")
    with pytest.raises(ValueError):
        BoundedExecutor(max_concurrency=1, max_queue=0, name="test")
