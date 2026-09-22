import pandas as pd
import pytest

from ml.training.baseline import train_logistic_regression_baseline
from ml.training.config import load_model_config


def test_logistic_regression_baseline_trains_and_predicts():
    config = load_model_config("configs/model.yaml")
    train_frame = pd.DataFrame(
        {
            "amount_log": [0.1, 0.2, 1.0, 1.2, 0.3, 1.4],
            "transaction_type": [
                "PAYMENT",
                "PAYMENT",
                "TRANSFER",
                "TRANSFER",
                "PAYMENT",
                "TRANSFER",
            ],
            "is_fraud": [0, 0, 1, 1, 0, 1],
        }
    )

    baseline = train_logistic_regression_baseline(
        train_frame,
        feature_columns=("amount_log", "transaction_type"),
        config=config,
    )

    serving_frame = train_frame.loc[:, ["amount_log", "transaction_type"]]
    probabilities = baseline.predict_proba(serving_frame)
    predictions = baseline.predict(serving_frame)
    assert probabilities.shape == (len(train_frame), 2)
    assert predictions.shape == (len(train_frame),)
    assert baseline.training_row_count == len(train_frame)
    assert baseline.target_distribution.positive_count == 3
    assert baseline.estimator.class_weight is None


def test_logistic_regression_training_requires_target_column():
    config = load_model_config("configs/model.yaml")
    feature_only_frame = pd.DataFrame(
        {
            "amount_log": [0.1, 1.0],
            "transaction_type": ["PAYMENT", "TRANSFER"],
        }
    )

    with pytest.raises(ValueError, match="Training target column not found"):
        train_logistic_regression_baseline(
            feature_only_frame,
            feature_columns=("amount_log", "transaction_type"),
            config=config,
        )
