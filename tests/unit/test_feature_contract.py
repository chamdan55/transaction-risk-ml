import pytest

from ml.contracts.features import (
    FEATURE_CONTRACT_VERSION,
    FORBIDDEN_MODEL_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    validate_model_feature_columns,
)
from ml.data.split import MODEL_DATASET_COLUMNS


def test_pre_transaction_contract_is_ordered_and_excludes_forbidden_features():
    assert FEATURE_CONTRACT_VERSION == "pre-transaction-v1"
    assert validate_model_feature_columns(MODEL_FEATURE_COLUMNS) == MODEL_FEATURE_COLUMNS
    assert not FORBIDDEN_MODEL_FEATURE_COLUMNS.intersection(MODEL_FEATURE_COLUMNS)


@pytest.mark.parametrize(
    "forbidden",
    [
        "origin_balance_after",
        "destination_balance_delta",
        "origin_balance_depleted",
        "transactions_last_1h",
        "is_fraud",
    ],
)
def test_contract_rejects_post_event_historical_and_label_features(forbidden):
    columns = list(MODEL_FEATURE_COLUMNS)
    columns[-1] = forbidden
    with pytest.raises(ValueError, match="exactly match"):
        validate_model_feature_columns(columns)


def test_model_dataset_projection_has_only_integrity_key_features_and_target():
    assert MODEL_DATASET_COLUMNS == ("transaction_id", *MODEL_FEATURE_COLUMNS, "is_fraud")
