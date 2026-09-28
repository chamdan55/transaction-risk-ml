from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from ml.contracts.features import MODEL_FEATURE_COLUMNS
from ml.monitoring.contracts import DELAYED_LABEL_SCHEMA_VERSION, DelayedLabel, PredictionEvent


def _features() -> dict[str, str | int | float]:
    return {
        name: ("PAYMENT" if name == "transaction_type" else 1.0) for name in MODEL_FEATURE_COLUMNS
    }


def test_prediction_event_contract_has_only_versioned_approved_features() -> None:
    event = PredictionEvent(
        feedback_id=uuid4(),
        event_time=datetime.now(UTC),
        model_name="random_forest",
        model_version="local-v1",
        feature_contract_version="pre-transaction-v1",
        features=_features(),
        risk_score=0.25,
        decision_threshold=0.5,
        decision="allow",
        latency_seconds=0.01,
    )

    assert event.schema_version == "prediction-event-v1"
    assert tuple(event.features) == tuple(_features())
    serialized = event.model_dump_json()
    assert "account_id" not in serialized
    assert "transaction_id" not in serialized
    assert "is_fraud" not in serialized


def test_prediction_event_rejects_unapproved_or_missing_features() -> None:
    fields = {
        "feedback_id": uuid4(),
        "event_time": datetime.now(UTC),
        "model_name": "random_forest",
        "model_version": "local-v1",
        "feature_contract_version": "pre-transaction-v1",
        "features": _features(),
        "risk_score": 0.25,
        "decision_threshold": 0.5,
        "decision": "allow",
        "latency_seconds": 0.01,
    }
    with pytest.raises(ValidationError, match="approved model feature contract"):
        PredictionEvent(**{**fields, "features": {**_features(), "origin_account_id": 42}})
    with pytest.raises(ValidationError, match="approved model feature contract"):
        PredictionEvent(**{**fields, "features": {"amount": 1.0}})
    with pytest.raises(ValidationError, match="approved transaction type"):
        PredictionEvent(
            **{
                **fields,
                "features": {**_features(), "transaction_type": "unrestricted user value"},
            }
        )


def test_delayed_label_contract_requires_timezone_and_binary_strict_integer() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        DelayedLabel(
            schema_version=DELAYED_LABEL_SCHEMA_VERSION,
            feedback_id=uuid4(),
            label=1,
            label_time=datetime(2026, 1, 1),
        )
    with pytest.raises(ValidationError):
        DelayedLabel(
            schema_version=DELAYED_LABEL_SCHEMA_VERSION,
            feedback_id=uuid4(),
            label=True,
            label_time=datetime.now(UTC),
        )
