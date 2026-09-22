"""Bounded in-process reliability primitives for synchronous predictions."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import TypeVar
from uuid import UUID

from app.api.schemas import PredictionResponse

T = TypeVar("T")


class PredictionBusyError(RuntimeError):
    """Raised when all configured prediction slots are occupied."""


class PredictionTimeoutError(RuntimeError):
    """Raised when prediction did not complete within its bounded timeout."""


class IdempotencyConflictError(RuntimeError):
    """Raised when an existing request ID is reused with a different payload."""


@dataclass
class _IdempotencyEntry:
    fingerprint: str
    expires_at: float
    result: asyncio.Future[PredictionResponse]


class InMemoryIdempotencyStore:
    """TTL-bounded idempotency for one API process, with no payload retention."""

    def __init__(self, *, ttl_seconds: float, max_entries: int) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._entries: dict[UUID, _IdempotencyEntry] = {}
        self._lock = asyncio.Lock()

    async def acquire(
        self, request_id: UUID, fingerprint: str
    ) -> tuple[bool, asyncio.Future[PredictionResponse]]:
        """Return whether this caller owns computation and the shared result future."""

        async with self._lock:
            now = monotonic()
            self._purge_expired(now)
            entry = self._entries.get(request_id)
            if entry is not None:
                if entry.fingerprint != fingerprint:
                    raise IdempotencyConflictError("request_id was reused with a different payload")
                return False, entry.result
            if len(self._entries) >= self._max_entries:
                self._evict_completed_entry()
            if len(self._entries) >= self._max_entries:
                raise PredictionBusyError("idempotency capacity is full")
            result: asyncio.Future[PredictionResponse] = asyncio.get_running_loop().create_future()
            self._entries[request_id] = _IdempotencyEntry(
                fingerprint=fingerprint,
                expires_at=now + self._ttl_seconds,
                result=result,
            )
            return True, result

    async def complete(self, request_id: UUID, response: PredictionResponse) -> None:
        async with self._lock:
            entry = self._entries.get(request_id)
            if entry is not None and not entry.result.done():
                entry.result.set_result(response)

    async def fail(self, request_id: UUID, _error: Exception) -> None:
        async with self._lock:
            entry = self._entries.pop(request_id, None)
            if entry is not None and not entry.result.done():
                entry.result.cancel()

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()

    def _purge_expired(self, now: float) -> None:
        expired = [
            request_id for request_id, entry in self._entries.items() if entry.expires_at <= now
        ]
        for request_id in expired:
            self._entries.pop(request_id)

    def _evict_completed_entry(self) -> None:
        for request_id, entry in self._entries.items():
            if entry.result.done():
                self._entries.pop(request_id)
                return


class PredictionExecutor:
    """Apply queue and execution bounds while keeping model access read-only."""

    def __init__(
        self,
        *,
        max_concurrent_predictions: int,
        queue_timeout_seconds: float,
        prediction_timeout_seconds: float,
    ) -> None:
        self._semaphore = asyncio.BoundedSemaphore(max_concurrent_predictions)
        self._queue_timeout_seconds = queue_timeout_seconds
        self._prediction_timeout_seconds = prediction_timeout_seconds
        self._running: set[asyncio.Task[object]] = set()

    async def run(self, operation: Callable[[], T]) -> T:
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self._queue_timeout_seconds)
        except TimeoutError as exc:
            raise PredictionBusyError("prediction concurrency limit reached") from exc

        task: asyncio.Task[T] = asyncio.create_task(asyncio.to_thread(operation))
        self._running.add(task)
        task.add_done_callback(self._running.discard)
        release_when_done = False
        try:
            return await asyncio.wait_for(
                asyncio.shield(task), timeout=self._prediction_timeout_seconds
            )
        except TimeoutError as exc:
            release_when_done = True
            task.add_done_callback(lambda _: self._semaphore.release())
            raise PredictionTimeoutError("prediction timed out") from exc
        finally:
            if not release_when_done:
                self._semaphore.release()

    async def shutdown(self, grace_seconds: float) -> None:
        """Give in-flight thread work a bounded period to finish during shutdown."""

        if self._running:
            await asyncio.wait(self._running, timeout=grace_seconds)
