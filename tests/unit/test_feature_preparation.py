from datetime import datetime

import pytest

from app.api.schemas import PredictionRequest, TransactionType
from app.services.feature_preparation import prepare_feature_frame
from ml.contracts.features import MODEL_FEATURE_COLUMNS


def test_prepare_feature_frame_matches_the_ordered_training_contract() -> None:
    frame = prepare_feature_frame(
        PredictionRequest(
            transaction_type=TransactionType.PAYMENT,
            amount=100.0,
            origin_balance_before=1_000.0,
            destination_balance_before=500.0,
            timestamp=datetime(2026, 1, 1, 10, 0, 0),
        )
    )

    assert tuple(frame.columns) == MODEL_FEATURE_COLUMNS
    assert frame.loc[0, "amount_log"] == pytest.approx(4.61512051684126)
    assert frame.loc[0, "amount_to_origin_balance_ratio"] == 0.1
    assert frame.loc[0, "amount_to_destination_balance_ratio"] == 0.2
    assert frame.loc[0, "transaction_day_of_week"] == 5
    assert frame.loc[0, "is_payment"] == 1


def test_prediction_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        PredictionRequest.model_validate(
            {
                "transaction_type": "PAYMENT",
                "amount": 100.0,
                "origin_balance_before": 1_000.0,
                "destination_balance_before": 500.0,
                "timestamp": "2026-01-01T10:00:00",
                "is_fraud": 1,
            }
        )
