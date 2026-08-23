"""Bounded process-local admission for expensive image inference requests."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator


class InferenceCapacityExceeded(RuntimeError):
    """Raised before image loading when the process has no free permit."""


class InferenceGateway:
    """Limit requests that may hold upload bytes or decoded images in memory."""

    def __init__(self, max_inflight_requests: int) -> None:
        if max_inflight_requests <= 0:
            raise ValueError("max_inflight_requests must be positive")
        self._max_inflight_requests = max_inflight_requests
        self._inflight_requests = 0

    @asynccontextmanager
    async def admit(self) -> AsyncIterator[None]:
        """Acquire one permit immediately and always return it on exit."""

        # FastAPI invokes this process-local service on one event loop. There
        # is no await between checking and incrementing, so another coroutine
        # cannot interleave with this critical section.
        if self._inflight_requests >= self._max_inflight_requests:
            raise InferenceCapacityExceeded
        self._inflight_requests += 1
        try:
            yield
        finally:
            self._inflight_requests -= 1
