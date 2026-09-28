"""Privacy-safe, per-application Prometheus runtime metrics."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Final

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from app.services.model_provider import ModelMetadata

LOGGER = logging.getLogger(__name__)

METRICS_CONTENT_TYPE: Final = "text/plain; version=0.0.4; charset=utf-8"
HTTP_DURATION_BUCKETS: Final = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)


class RuntimeMetrics:
    """Expose only bounded operational labels; never accept request data as a label."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self.http_requests = Counter(
            "trm_http_requests_total",
            "Completed HTTP requests.",
            ("method", "path", "status"),
            registry=self.registry,
        )
        self.http_duration = Histogram(
            "trm_http_request_duration_seconds",
            "End-to-end HTTP request duration in seconds.",
            ("method", "path"),
            buckets=HTTP_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.http_in_flight = Gauge(
            "trm_http_in_flight_requests",
            "HTTP requests currently being processed.",
            ("method", "path"),
            registry=self.registry,
        )
        self.validation_rejections = Counter(
            "trm_validation_rejections_total",
            "Requests rejected by the public request contract.",
            ("path",),
            registry=self.registry,
        )
        self.model_loads = Counter(
            "trm_model_load_total",
            "Model loading attempts by outcome.",
            ("outcome",),
            registry=self.registry,
        )
        self.model_load_duration = Histogram(
            "trm_model_load_duration_seconds",
            "Model loading duration in seconds.",
            buckets=HTTP_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.model_ready = Gauge(
            "trm_model_ready",
            "One when the configured model is ready to score, otherwise zero.",
            registry=self.registry,
        )
        self.model_info = Gauge(
            "trm_model_info",
            "Metadata for the currently configured model; exactly one active series is one.",
            ("model_name", "model_version", "model_source", "feature_contract_version"),
            registry=self.registry,
        )
        self.prediction_decisions = Counter(
            "trm_prediction_decisions_total",
            "Successful prediction decisions by bounded decision value.",
            ("decision",),
            registry=self.registry,
        )
        self.prediction_events = Counter(
            "trm_prediction_events_total",
            "Prediction events accepted by or dropped from the bounded background queue.",
            ("outcome",),
            registry=self.registry,
        )
        self.prediction_event_store_failures = Counter(
            "trm_prediction_event_store_failures_total",
            "Background prediction-event persistence attempts that failed and will retry.",
            registry=self.registry,
        )
        self.delayed_label_ingestions = Counter(
            "trm_delayed_label_ingestions_total",
            "Delayed-label submissions by bounded outcome.",
            ("outcome",),
            registry=self.registry,
        )

    def observe_request(self, method: str, path: str, status_code: int, started_at: float) -> None:
        self._safe(self.http_requests.labels(method=method, path=path, status=str(status_code)).inc)
        self._safe(
            self.http_duration.labels(method=method, path=path).observe, perf_counter() - started_at
        )

    def begin_request(self, method: str, path: str) -> None:
        self._safe(self.http_in_flight.labels(method=method, path=path).inc)

    def end_request(self, method: str, path: str) -> None:
        self._safe(self.http_in_flight.labels(method=method, path=path).dec)

    def record_validation_rejection(self, path: str) -> None:
        self._safe(self.validation_rejections.labels(path=path).inc)

    def record_model_load(self, *, ready: bool, duration_seconds: float) -> None:
        self._safe(self.model_loads.labels(outcome="success" if ready else "failure").inc)
        self._safe(self.model_load_duration.observe, duration_seconds)
        self._safe(self.model_ready.set, 1 if ready else 0)

    def set_model_metadata(self, metadata: ModelMetadata | None) -> None:
        if metadata is None:
            self._safe(self.model_ready.set, 0)
            return
        self._safe(self.model_ready.set, 1)
        self._safe(
            self.model_info.labels(
                model_name=metadata.model_name,
                model_version=metadata.model_version,
                model_source=metadata.model_source,
                feature_contract_version=metadata.feature_contract_version,
            ).set,
            1,
        )

    def record_decision(self, decision: str) -> None:
        if decision not in {"allow", "review"}:
            LOGGER.warning("Ignoring unknown prediction decision metric")
            return
        self._safe(self.prediction_decisions.labels(decision=decision).inc)

    def record_prediction_event(self, outcome: str, *, count: int = 1) -> None:
        if outcome not in {"queued", "dropped"}:
            return
        self._safe(self.prediction_events.labels(outcome=outcome).inc, count)

    def record_prediction_event_store_failure(self) -> None:
        self._safe(self.prediction_event_store_failures.inc)

    def record_delayed_label(self, outcome: str) -> None:
        if outcome not in {"accepted", "duplicate", "conflict", "unavailable"}:
            return
        self._safe(self.delayed_label_ingestions.labels(outcome=outcome).inc)

    def render(self) -> bytes:
        try:
            return generate_latest(self.registry)
        except Exception:  # pragma: no cover - defensive telemetry boundary
            LOGGER.exception("Unable to render Prometheus metrics")
            return b""

    @staticmethod
    def _safe(operation, *args) -> None:  # type: ignore[no-untyped-def]
        """Metrics must never change prediction availability."""

        try:
            operation(*args)
        except Exception:  # pragma: no cover - prometheus client defensive boundary
            LOGGER.exception("Unable to update runtime metric")
