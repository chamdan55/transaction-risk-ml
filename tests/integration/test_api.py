from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.model_provider import ModelMetadata


class ReadyProvider:
    """Deterministic in-memory stand-in for HTTP contract verification."""

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

    def predict_score(self, feature_frame):
        assert tuple(feature_frame.columns)[0] == "transaction_type"
        return 0.75


def _payload() -> dict[str, object]:
    return {
        "transaction_type": "PAYMENT",
        "amount": 100.0,
        "origin_balance_before": 1_000.0,
        "destination_balance_before": 500.0,
        "timestamp": "2026-01-01T10:00:00",
        "request_id": "3b241101-e2bb-4255-8caf-4136c566a962",
    }


def test_prediction_contract_returns_decision_and_model_metadata() -> None:
    app = create_app(provider=ReadyProvider())
    with TestClient(app) as client:
        response = client.post("/v1/predictions", json=_payload())

    assert response.status_code == 200
    assert response.json() == {
        "request_id": "3b241101-e2bb-4255-8caf-4136c566a962",
        "risk_score": 0.75,
        "decision": "review",
        "model_name": "random_forest",
        "model_version": "test-v1",
        "model_source": "local",
        "threshold": 0.5,
        "threshold_policy": "business_cost",
        "feature_contract_version": "pre-transaction-v1",
    }


def test_liveness_stays_healthy_while_a_missing_model_is_unready(tmp_path) -> None:
    app = create_app(
        Settings(
            model_artifact_path=tmp_path / "missing.joblib",
            model_evaluation_report_path=tmp_path / "missing.json",
        )
    )
    with TestClient(app) as client:
        assert client.get("/health/live").json() == {"status": "live"}
        ready_response = client.get("/health/ready")
        prediction_response = client.post("/v1/predictions", json=_payload())

    assert ready_response.status_code == 503
    assert ready_response.json()["detail"]["status"] == "not_ready"
    assert prediction_response.status_code == 503


def test_prediction_rejects_unknown_request_fields() -> None:
    app = create_app(provider=ReadyProvider())
    payload = {**_payload(), "is_fraud": 1}
    with TestClient(app) as client:
        response = client.post("/v1/predictions", json=payload)

    assert response.status_code == 422
