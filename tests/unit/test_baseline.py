import pandas as pd

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

    probabilities = baseline.predict_proba(train_frame)
    predictions = baseline.predict(train_frame)
    assert probabilities.shape == (len(train_frame), 2)
    assert predictions.shape == (len(train_frame),)
    assert baseline.training_row_count == len(train_frame)
    assert baseline.target_distribution.positive_count == 3
    assert baseline.estimator.class_weight == {0: 1.0, 1: 1.0}
