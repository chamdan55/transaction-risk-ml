import importlib.util

import pandas as pd
import pytest

from ml.training.config import load_model_config
from ml.training.models import train_random_forest, train_xgboost

FEATURE_COLUMNS = ("amount_log", "transaction_type")


@pytest.fixture
def train_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount_log": [0.1, 0.2, 1.0, 1.2, 0.3, 1.4, 0.4, 1.6],
            "transaction_type": [
                "PAYMENT",
                "PAYMENT",
                "TRANSFER",
                "TRANSFER",
                "PAYMENT",
                "TRANSFER",
                "PAYMENT",
                "TRANSFER",
            ],
            "is_fraud": [0, 0, 1, 1, 0, 1, 0, 1],
        }
    )


def test_random_forest_trains_and_predicts(train_frame):
    config = load_model_config("configs/model.yaml")
    config.model_params["random_forest"]["n_jobs"] = 1
    model = train_random_forest(
        train_frame,
        feature_columns=FEATURE_COLUMNS,
        config=config,
    )

    assert model.model_name == "random_forest"
    serving_frame = train_frame.loc[:, list(FEATURE_COLUMNS)]
    assert model.predict_proba(serving_frame).shape == (len(train_frame), 2)
    assert model.predict(serving_frame).shape == (len(train_frame),)


@pytest.mark.skipif(
    importlib.util.find_spec("xgboost") is None,
    reason="xgboost is not installed in the active environment",
)
def test_xgboost_trains_and_predicts(train_frame):
    config = load_model_config("configs/model.yaml")
    model = train_xgboost(
        train_frame,
        feature_columns=FEATURE_COLUMNS,
        config=config,
    )

    assert model.model_name == "xgboost"
    serving_frame = train_frame.loc[:, list(FEATURE_COLUMNS)]
    assert model.predict_proba(serving_frame).shape == (len(train_frame), 2)
    assert model.predict(serving_frame).shape == (len(train_frame),)
