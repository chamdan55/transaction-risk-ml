import numpy as np
import pandas as pd
import pytest

from ml.training.preprocessing import (
    FeaturePreparationError,
    FeaturePreprocessor,
    prepare_feature_frame,
)

FEATURE_COLUMNS = ("amount_log", "transaction_type")


@pytest.fixture
def train_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount_log": [1.0, 2.0, np.nan],
            "transaction_type": ["PAYMENT", "TRANSFER", "PAYMENT"],
            "is_fraud": [0, 1, 0],
        }
    )


def test_prepare_feature_frame_excludes_target_and_preserves_feature_order(
    train_frame,
):
    prepared = prepare_feature_frame(
        train_frame,
        feature_columns=FEATURE_COLUMNS,
        target_column="is_fraud",
    )

    assert list(prepared.features.columns) == list(FEATURE_COLUMNS)
    assert prepared.target.tolist() == [0, 1, 0]


def test_preprocessor_fits_train_and_handles_unseen_categories(train_frame):
    preprocessor = FeaturePreprocessor(FEATURE_COLUMNS, "is_fraud")
    train_matrix = preprocessor.fit_transform(train_frame)
    validation_frame = pd.DataFrame(
        {
            "amount_log": [3.0],
            "transaction_type": ["CASH_OUT"],
        }
    )
    validation_matrix = preprocessor.transform(validation_frame)

    assert preprocessor.numeric_columns == ("amount_log",)
    assert preprocessor.categorical_columns == ("transaction_type",)
    assert train_matrix.shape[1] == validation_matrix.shape[1]
    assert len(preprocessor.get_feature_names_out()) == train_matrix.shape[1]


def test_preprocessor_requires_fit_before_transform(train_frame):
    preprocessor = FeaturePreprocessor(FEATURE_COLUMNS, "is_fraud")

    with pytest.raises(FeaturePreparationError, match="must be fitted"):
        preprocessor.transform(train_frame)


def test_preprocessor_rejects_missing_columns(train_frame):
    preprocessor = FeaturePreprocessor(FEATURE_COLUMNS, "is_fraud")

    with pytest.raises(FeaturePreparationError, match="missing required columns"):
        preprocessor.fit(train_frame.drop(columns=["transaction_type"]))


def test_prepare_feature_frame_requires_target_only_for_training_data(train_frame):
    with pytest.raises(FeaturePreparationError, match="required target column"):
        prepare_feature_frame(
            train_frame.drop(columns=["is_fraud"]),
            feature_columns=FEATURE_COLUMNS,
            target_column="is_fraud",
        )
