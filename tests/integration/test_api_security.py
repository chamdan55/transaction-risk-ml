from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.model_provider import ModelMetadata


class CountingProvider:
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

    def __init__(self, *, delay_seconds: float = 0) -> None:
        self.delay_seconds = delay_seconds
        self.call_count = 0
        self.active_calls = 0
        self.max_active_calls = 0
        self._lock = Lock()

    def load(self) -> None:
        return None

    def predict_score(self, _feature_frame) -> float:
        with self._lock:
            self.call_count += 1
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            if self.delay_seconds:
                time.sleep(self.delay_seconds)
            return 0.75
        finally:
            with self._lock:
                self.active_calls -= 1


def _payload(request_id: str = "3b241101-e2bb-4255-8caf-4136c566a962") -> dict[str, object]:
    return {
        "transaction_type": "PAYMENT",
        "amount": 100.0,
        "origin_balance_before": 1_000.0,
        "destination_balance_before": 500.0,
        "timestamp": "2026-01-01T10:00:00",
        "request_id": request_id,
    }


def test_secured_mode_rejects_unauthorized_prediction_but_keeps_health_public() -> None:
    app = create_app(
        Settings(api_auth_enabled=True, api_key="not-in-response"),
        provider=CountingProvider(),
    )
    with TestClient(app) as client:
        health_response = client.get("/health/live")
        response = client.post("/v1/predictions", json=_payload())
        authorized_response = client.post(
            "/v1/predictions", json=_payload(), headers={"X-API-Key": "not-in-response"}
        )
        model_info_response = client.get("/model/info", headers={"X-API-Key": "not-in-response"})

    assert health_response.status_code == 200
    assert response.status_code == 401
    assert "not-in-response" not in response.text
    assert authorized_response.status_code == 200
    assert model_info_response.status_code == 200


def test_prediction_request_id_is_idempotent_and_conflicts_on_different_payload() -> None:
    provider = CountingProvider()
    app = create_app(provider=provider)
    with TestClient(app) as client:
        first = client.post("/v1/predictions", json=_payload())
        replay = client.post("/v1/predictions", json=_payload())
        conflict = client.post("/v1/predictions", json={**_payload(), "amount": 101.0})

    assert first.status_code == 200
    assert replay.status_code == 200
    assert replay.headers["X-Idempotent-Replay"] == "true"
    assert replay.json() == first.json()
    assert conflict.status_code == 409
    assert provider.call_count == 1


def test_oversized_request_and_timeout_have_safe_statuses() -> None:
    app = create_app(
        Settings(max_request_bytes=1_024, prediction_timeout_seconds=0.01),
        provider=CountingProvider(delay_seconds=0.05),
    )
    oversized_payload = {**_payload(), "ignored": "x" * 2_000}
    with TestClient(app) as client:
        oversized = client.post("/v1/predictions", json=oversized_payload)
        timed_out = client.post("/v1/predictions", json=_payload())

    assert oversized.status_code == 413
    assert timed_out.status_code == 504
    assert "ignored" not in oversized.text


def test_concurrent_predictions_respect_the_configured_model_limit() -> None:
    provider = CountingProvider(delay_seconds=0.03)
    app = create_app(
        Settings(max_concurrent_predictions=2, prediction_queue_timeout_seconds=0.5),
        provider=provider,
    )
    with TestClient(app) as client:
        with ThreadPoolExecutor(max_workers=4) as executor:
            responses = list(
                executor.map(
                    lambda index: client.post(
                        "/v1/predictions",
                        json=_payload(f"3b241101-e2bb-4255-8caf-4136c566a9{index:02}"),
                    ),
                    range(4),
                )
            )

    assert [response.status_code for response in responses] == [200, 200, 200, 200]
    assert provider.max_active_calls <= 2


def test_openapi_exposes_prediction_contract_without_the_removed_config_route() -> None:
    app = create_app(provider=CountingProvider())
    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()
        config_response = client.get("/config")

    assert "/v1/predictions" in schema["paths"]
    assert "/config" not in schema["paths"]
    assert config_response.status_code == 404


def test_validation_error_does_not_reflect_account_identifier_or_payload(caplog) -> None:
    account_identifier = "origin-account-must-not-leak"
    app = create_app(provider=CountingProvider())
    with caplog.at_level(logging.INFO):
        with TestClient(app) as client:
            response = client.post(
                "/v1/predictions",
                json={**_payload(), "origin_account_id": account_identifier},
            )

    assert response.status_code == 422
    assert account_identifier not in response.text
    assert account_identifier not in caplog.text
