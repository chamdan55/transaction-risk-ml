from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.metrics import RuntimeMetrics
from app.services.model_provider import ModelMetadata


class ReadyProvider:
    is_ready = True
    load_error = None
    metadata = ModelMetadata(
        model_name="random_forest",
        model_version="test-v1",
        model_source="local",
        threshold=0.5,
        threshold_policy="business_cost",
        feature_contract_version="pre-transaction-v1",
    )

    def load(self) -> None:
        return None

    def predict_score(self, _feature_frame) -> float:
        return 0.75


class UnreadyProvider:
    is_ready = False
    load_error = "sanitized load failure"

    @property
    def metadata(self) -> ModelMetadata:
        raise RuntimeError("not ready")

    def load(self) -> None:
        return None


def _payload() -> dict[str, object]:
    return {
        "transaction_type": "PAYMENT",
        "amount": 100.0,
        "origin_balance_before": 1_000.0,
        "destination_balance_before": 500.0,
        "timestamp": "2026-01-01T10:00:00",
        "request_id": "3b241101-e2bb-4255-8caf-4136c566a962",
    }


def test_metrics_include_bounded_operational_labels_and_no_request_identifiers() -> None:
    account_identifier = "account-id-must-not-be-a-metric-label"
    app = create_app(provider=ReadyProvider())
    with TestClient(app) as client:
        assert client.post("/v1/predictions", json=_payload()).status_code == 200
        assert (
            client.post(
                "/v1/predictions",
                json={**_payload(), "origin_account_id": account_identifier},
            ).status_code
            == 422
        )
        metrics = client.get("/metrics")

    assert metrics.status_code == 200
    assert metrics.headers["content-type"].startswith("text/plain")
    assert (
        'trm_http_requests_total{method="POST",path="/v1/predictions",status="200"}' in metrics.text
    )
    assert 'trm_validation_rejections_total{path="/v1/predictions"}' in metrics.text
    assert 'trm_prediction_decisions_total{decision="review"}' in metrics.text
    assert 'trm_prediction_events_total{outcome="queued"}' in metrics.text
    assert (
        'trm_model_info{feature_contract_version="pre-transaction-v1",model_name="random_forest",model_source="local",model_version="test-v1"}'
        in metrics.text
    )
    assert account_identifier not in metrics.text
    assert "3b241101-e2bb-4255-8caf-4136c566a962" not in metrics.text
    assert 'path="/metrics"' not in metrics.text


def test_failed_model_load_is_visible_without_making_liveness_fail() -> None:
    app = create_app(Settings(), provider=UnreadyProvider())
    with TestClient(app) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")
        metrics = client.get("/metrics")

    assert live.status_code == 200
    assert ready.status_code == 503
    assert 'trm_model_load_total{outcome="failure"} 1.0' in metrics.text
    assert "trm_model_ready 0.0" in metrics.text


def test_telemetry_update_failure_is_isolated_from_callers() -> None:
    metrics = RuntimeMetrics()

    def broken_operation() -> None:
        raise RuntimeError("simulated telemetry outage")

    metrics._safe(broken_operation)
