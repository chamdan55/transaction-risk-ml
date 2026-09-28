"""Bounded asynchronous prediction-event intake with a local SQLite store."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ml.monitoring.contracts import DelayedLabel, PredictionEvent

LOGGER = logging.getLogger(__name__)


class LabelConflictError(ValueError):
    """A feedback ID already has a different final label."""


class PredictionEventStore:
    """Queue successful prediction events and persist them away from request handlers.

    The in-memory queue is intentionally bounded. Events are at-least-once while the process is
    alive (SQLite writes are idempotent by feedback ID), with retry on storage errors. Events still
    queued at process termination or rejected because the queue is full can be lost; these cases
    are counted and logged without changing inference availability.
    """

    def __init__(
        self,
        database_path: str | Path,
        *,
        queue_maxsize: int = 5_000,
        retention_days: int = 90,
        on_write_failure: Callable[[], None] | None = None,
        on_shutdown_drop: Callable[[int], None] | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self._queue: asyncio.Queue[PredictionEvent] = asyncio.Queue(maxsize=queue_maxsize)
        self.retention_days = retention_days
        self._worker: asyncio.Task[None] | None = None
        self._initialization_lock = threading.Lock()
        self._initialized = False
        self._on_write_failure = on_write_failure
        self._on_shutdown_drop = on_shutdown_drop
        self._in_flight_events = 0
        self.dropped_events = 0
        self.write_failures = 0

    def start(self) -> None:
        """Start the background writer without making API readiness depend on SQLite."""

        if self._worker is None:
            self._worker = asyncio.create_task(self._run_writer(), name="prediction-event-writer")

    def enqueue(self, event: PredictionEvent) -> bool:
        """Add an event without awaiting disk/network I/O."""

        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self.dropped_events += 1
            return False
        return True

    async def record_label(self, label: DelayedLabel) -> bool:
        """Persist a delayed label; return True when it is an idempotent replay."""

        return await asyncio.to_thread(self._record_label_sync, label)

    async def close(self, grace_seconds: float) -> None:
        """Drain for a bounded period, then cancel the writer if storage remains unavailable."""

        if self._worker is None:
            return
        try:
            await asyncio.wait_for(self._queue.join(), timeout=grace_seconds)
        except TimeoutError:
            pending = self._queue.qsize() + self._in_flight_events
            self.dropped_events += pending
            if pending and self._on_shutdown_drop is not None:
                self._on_shutdown_drop(pending)
            LOGGER.warning(
                "Prediction event writer shutdown grace expired; dropped_events=%s",
                pending,
            )
        self._worker.cancel()
        await asyncio.gather(self._worker, return_exceptions=True)
        self._worker = None

    def _connect(self) -> sqlite3.Connection:
        self._ensure_initialized()
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._initialization_lock:
            if self._initialized:
                return
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.database_path, timeout=5)
            try:
                with connection:
                    connection.execute("PRAGMA busy_timeout = 5000")
                    connection.execute("PRAGMA journal_mode = WAL")
                    connection.executescript(
                        """
                        CREATE TABLE IF NOT EXISTS prediction_events (
                            feedback_id TEXT PRIMARY KEY,
                            schema_version TEXT NOT NULL,
                            event_time TEXT NOT NULL,
                            model_name TEXT NOT NULL,
                            model_version TEXT NOT NULL,
                            feature_contract_version TEXT NOT NULL,
                            features_json TEXT NOT NULL,
                            risk_score REAL NOT NULL,
                            decision_threshold REAL NOT NULL,
                            decision TEXT NOT NULL,
                            latency_seconds REAL NOT NULL,
                            stored_at TEXT NOT NULL
                        );
                        CREATE INDEX IF NOT EXISTS ix_prediction_events_time
                            ON prediction_events(event_time);
                        CREATE INDEX IF NOT EXISTS ix_prediction_events_segment
                            ON prediction_events(model_name, model_version, feature_contract_version);
                        CREATE TABLE IF NOT EXISTS delayed_labels (
                            feedback_id TEXT PRIMARY KEY,
                            schema_version TEXT NOT NULL,
                            label INTEGER NOT NULL CHECK (label IN (0, 1)),
                            label_time TEXT NOT NULL,
                            stored_at TEXT NOT NULL
                        );
                        CREATE INDEX IF NOT EXISTS ix_delayed_labels_time
                            ON delayed_labels(label_time);
                        """
                    )
                    columns = {
                        row[1] for row in connection.execute("PRAGMA table_info(prediction_events)")
                    }
                    if "decision_threshold" not in columns:
                        connection.execute(
                            "ALTER TABLE prediction_events "
                            "ADD COLUMN decision_threshold REAL NOT NULL DEFAULT 0.5"
                        )
            finally:
                connection.close()
            self._initialized = True

    async def _run_writer(self) -> None:
        while True:
            first = await self._queue.get()
            batch = [first]
            while len(batch) < 100:
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            self._in_flight_events = len(batch)
            delay_seconds = 0.25
            while True:
                try:
                    await asyncio.to_thread(self._persist_events_sync, batch)
                    break
                except asyncio.CancelledError:
                    for _ in batch:
                        self._queue.task_done()
                    self._in_flight_events = 0
                    raise
                except Exception:
                    self.write_failures += 1
                    if self._on_write_failure is not None:
                        self._on_write_failure()
                    LOGGER.warning(
                        "Unable to persist prediction-event batch; retrying", exc_info=True
                    )
                    await asyncio.sleep(delay_seconds)
                    delay_seconds = min(delay_seconds * 2, 5.0)
            for _ in batch:
                self._queue.task_done()
            self._in_flight_events = 0

    def _persist_events_sync(self, events: list[PredictionEvent]) -> None:
        now = datetime.now(UTC).isoformat()
        rows = [
            (
                str(event.feedback_id),
                event.schema_version,
                event.event_time.astimezone(UTC).isoformat(),
                event.model_name,
                event.model_version,
                event.feature_contract_version,
                json.dumps(event.features, sort_keys=True, separators=(",", ":")),
                event.risk_score,
                event.decision_threshold,
                event.decision,
                event.latency_seconds,
                now,
            )
            for event in events
        ]
        with self._connection() as connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO prediction_events (
                    feedback_id, schema_version, event_time, model_name, model_version,
                    feature_contract_version, features_json, risk_score, decision_threshold,
                    decision, latency_seconds, stored_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._purge_expired(connection, self.retention_days)

    def _record_label_sync(self, label: DelayedLabel) -> bool:
        feedback_id = str(label.feedback_id)
        label_time = label.label_time.astimezone(UTC).isoformat()
        with self._connection() as connection:
            # Serialize the read/compare/write sequence so simultaneous retries cannot
            # both observe a missing row and turn the loser into an IntegrityError/503.
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT label FROM delayed_labels WHERE feedback_id = ?", (feedback_id,)
            ).fetchone()
            if existing is not None:
                if int(existing["label"]) != label.label:
                    raise LabelConflictError("feedback_id already has a different label")
                return True
            connection.execute(
                """
                INSERT INTO delayed_labels (feedback_id, schema_version, label, label_time, stored_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    feedback_id,
                    label.schema_version,
                    label.label,
                    label_time,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return False

    def read_snapshot(
        self,
        *,
        window_days: int,
        retention_days: int,
        label_grace_days: int,
        max_events: int,
    ) -> dict[str, Any]:
        """Purge expired rows and return bounded, joined rows plus operational counts."""

        now = datetime.now(UTC)
        retention_cutoff = (now - timedelta(days=retention_days)).isoformat()
        window_cutoff = (now - timedelta(days=window_days)).isoformat()
        late_cutoff = (now - timedelta(days=label_grace_days)).isoformat()
        with self._connection() as connection:
            self._purge_expired(connection, retention_days, cutoff=retention_cutoff)
            connection.commit()
            rows = connection.execute(
                """
                SELECT p.*, l.label, l.label_time
                FROM prediction_events AS p
                LEFT JOIN delayed_labels AS l
                  ON l.feedback_id = p.feedback_id AND l.label_time >= p.event_time
                WHERE p.event_time >= ?
                ORDER BY p.event_time DESC
                LIMIT ?
                """,
                (window_cutoff, max_events),
            ).fetchall()
            unmatched = connection.execute(
                """
                SELECT COUNT(*) FROM delayed_labels AS l
                LEFT JOIN prediction_events AS p ON p.feedback_id = l.feedback_id
                WHERE l.label_time >= ? AND (p.feedback_id IS NULL OR l.label_time < p.event_time)
                """,
                (retention_cutoff,),
            ).fetchone()[0]
            overdue = connection.execute(
                """
                SELECT COUNT(*) FROM prediction_events AS p
                LEFT JOIN delayed_labels AS l
                  ON l.feedback_id = p.feedback_id AND l.label_time >= p.event_time
                WHERE p.event_time < ? AND l.feedback_id IS NULL
                """,
                (late_cutoff,),
            ).fetchone()[0]
            event_count = connection.execute(
                "SELECT COUNT(*) FROM prediction_events WHERE event_time >= ?", (window_cutoff,)
            ).fetchone()[0]
            label_count = connection.execute(
                """
                SELECT COUNT(*) FROM delayed_labels AS l
                JOIN prediction_events AS p ON p.feedback_id = l.feedback_id
                WHERE p.event_time >= ? AND l.label_time >= p.event_time
                """,
                (window_cutoff,),
            ).fetchone()[0]

        events = []
        for row in reversed(rows):
            item = dict(row)
            item["features"] = json.loads(item.pop("features_json"))
            item["label"] = int(item["label"]) if item["label"] is not None else None
            events.append(item)
        return {
            "events": events,
            "event_count": int(event_count),
            "joined_label_count": int(label_count),
            "unmatched_label_count": int(unmatched),
            "late_unlabeled_count": int(overdue),
            "truncated": int(event_count) > len(events),
            "window_start": window_cutoff,
            "retention_start": retention_cutoff,
        }

    @staticmethod
    def _purge_expired(
        connection: sqlite3.Connection,
        retention_days: int,
        *,
        cutoff: str | None = None,
    ) -> None:
        retention_cutoff = (
            cutoff or (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
        )
        connection.execute("DELETE FROM delayed_labels WHERE label_time < ?", (retention_cutoff,))
        connection.execute(
            "DELETE FROM prediction_events WHERE event_time < ?", (retention_cutoff,)
        )
