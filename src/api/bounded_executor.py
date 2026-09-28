"""Bounded async bridge for synchronous model work."""

from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


class ExecutionQueueFull(RuntimeError):
    """Raised when executor capacity is saturated."""


class ExecutorClosed(RuntimeError):
    """Raised when work is submitted after shutdown."""


class BoundedExecutor:
    """Run blocking callables outside the event loop with bounded queueing."""

    def __init__(self, max_concurrency: int, max_queue: int, name: str) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if max_queue < 1:
            raise ValueError("max_queue must be at least 1")

        self.max_concurrency = max_concurrency
        self.max_queue = max_queue
        self.name = name
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._pool = ThreadPoolExecutor(
            max_workers=max_concurrency,
            thread_name_prefix=f"signrr-{name}",
        )
        self._workers: list[asyncio.Task] = []
        self._started = False
        self._closed = False

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        if self._closed:
            raise ExecutorClosed(f"{self.name} executor is closed")
        if self._started:
            return
        self._workers = [
            asyncio.create_task(self._worker(), name=f"{self.name}-worker-{index}")
            for index in range(self.max_concurrency)
        ]
        self._started = True

    async def submit(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if self._closed:
            raise ExecutorClosed(f"{self.name} executor is closed")
        if not self._started:
            raise RuntimeError(f"{self.name} executor has not been started")

        loop = asyncio.get_running_loop()
        result_future = loop.create_future()
        item = (result_future, func, args, kwargs)

        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull as exc:
            raise ExecutionQueueFull(f"{self.name} execution queue is full") from exc

        try:
            return await result_future
        except asyncio.CancelledError:
            result_future.cancel()
            raise

    async def _worker(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            result_future, func, args, kwargs = await self._queue.get()
            try:
                if result_future.cancelled():
                    continue

                call = functools.partial(func, *args, **kwargs)
                try:
                    result = await loop.run_in_executor(self._pool, call)
                except BaseException as exc:
                    if not result_future.cancelled():
                        result_future.set_exception(exc)
                else:
                    if not result_future.cancelled():
                        result_future.set_result(result)
            finally:
                self._queue.task_done()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        while True:
            try:
                result_future, _func, _args, _kwargs = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                result_future.cancel()
                self._queue.task_done()

        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self._pool.shutdown(wait=False, cancel_futures=True)
