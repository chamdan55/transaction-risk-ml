from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from uuid import UUID

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.model_provider import ModelMetadata
from ml.monitoring.contracts import DelayedLabel
from ml.monitoring.store import PredictionEventStore


class ReadyProvider:
    is_ready = True
    load_error = None
    metadata = ModelMetadata(
        model_name="random_forest",
        model_version="feedback-test-v1",
        model_source="local",
        threshold=0.5,
        threshold_policy="business_cost",
        feature_contract_version="pre-transaction-v1",
    )

    def load(self) -> None:
        return None

    def predict_score(self, _feature_frame) -> float:
        return 0.75


def _payload() -> dict[str, object]:
    return {
        "transaction_type": "PAYMENT",
        "amount": 100.0,
        "origin_balance_before": 1_000.0,
        "destination_balance_before": 500.0,
        "timestamp": "2026-01-01T10:00:00Z",
    }


def test_prediction_event_and_delayed_label_are_persisted_and_joinable(tmp_path) -> None:
    database = tmp_path / "monitoring" / "events.sqlite3"
    app = create_app(
        Settings(ml_monitoring_database_path=database),
        provider=ReadyProvider(),
    )
    with TestClient(app) as client:
        prediction = client.post("/v1/predictions", json=_payload())
        assert prediction.status_code == 200
        feedback_id = UUID(prediction.json()["feedback_id"])
        label_payload = {
            "schema_version": "delayed-label-v1",
            "feedback_id": str(feedback_id),
            "label": 1,
            "label_time": (datetime.now(UTC) + timedelta(seconds=1)).isoformat(),
        }
        accepted = client.post("/v1/feedback/labels", json=label_payload)
        duplicate = client.post("/v1/feedback/labels", json=label_payload)
        conflict = client.post("/v1/feedback/labels", json={**label_payload, "label": 0})

    assert accepted.status_code == 202
    assert accepted.json()["status"] == "accepted"
    assert duplicate.status_code == 202
    assert duplicate.json()["status"] == "duplicate"
    assert conflict.status_code == 409

    snapshot = PredictionEventStore(database).read_snapshot(
        window_days=30,
        retention_days=90,
        label_grace_days=7,
        max_events=100,
    )
    assert snapshot["event_count"] == 1
    assert snapshot["joined_label_count"] == 1
    assert snapshot["events"][0]["feedback_id"] == str(feedback_id)
    assert "request_id" not in snapshot["events"][0]


def test_prediction_is_not_failed_when_event_queue_is_unavailable() -> None:
    class BrokenEventStore:
        def start(self) -> None:
            pass

        def enqueue(self, _event) -> bool:
            raise OSError("simulated monitoring outage")

        async def close(self, _grace_seconds: float) -> None:
            pass

    app = create_app(provider=ReadyProvider(), event_store=BrokenEventStore())
    with TestClient(app) as client:
        response = client.post("/v1/predictions", json=_payload())
        ready = client.get("/health/ready")

    assert response.status_code == 200
    assert ready.status_code == 200


def test_label_contract_rejects_extra_raw_account_fields(tmp_path) -> None:
    app = create_app(
        Settings(ml_monitoring_database_path=tmp_path / "events.sqlite3"),
        provider=ReadyProvider(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/feedback/labels",
            json={
                "schema_version": "delayed-label-v1",
                "feedback_id": "2c7a1496-4d15-4a8f-88e7-c94d20d4c239",
                "label": 1,
                "label_time": datetime.now(UTC).isoformat(),
                "origin_account_id": "must-not-be-retained",
            },
        )

    assert response.status_code == 422
    assert "must-not-be-retained" not in response.text


def test_concurrent_same_label_retries_are_idempotent(tmp_path) -> None:
    store = PredictionEventStore(tmp_path / "events.sqlite3")
    payload = {
        "schema_version": "delayed-label-v1",
        "feedback_id": UUID("2c7a1496-4d15-4a8f-88e7-c94d20d4c239"),
        "label": 1,
        "label_time": datetime.now(UTC),
    }
    label = DelayedLabel(**payload)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(store._record_label_sync, [label] * 16))

    assert results.count(False) == 1
    assert results.count(True) == 15
