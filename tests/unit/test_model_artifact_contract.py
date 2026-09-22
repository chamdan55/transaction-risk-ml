from types import SimpleNamespace

import pandas as pd
import pytest

from ml.contracts.features import MODEL_FEATURE_COLUMNS
from ml.training.config import load_model_config
from pipelines.train_models import (
    _build_serving_input_example,
    _save_artifacts,
    _validate_logged_model_serving_input,
)


def test_saved_model_artifact_requires_the_prediction_feature_contract(tmp_path):
    config = load_model_config("configs/model.yaml")
    config = config.__class__(
        **{
            **config.__dict__,
            "model_directory": tmp_path / "models",
        }
    )
    invalid_model = SimpleNamespace(
        preprocessor=SimpleNamespace(feature_columns=("origin_balance_after",))
    )

    with pytest.raises(ValueError, match="approved feature contract"):
        _save_artifacts({"invalid": invalid_model}, config)


def test_saved_model_artifact_accepts_the_prediction_feature_contract(tmp_path):
    config = load_model_config("configs/model.yaml")
    config = config.__class__(
        **{
            **config.__dict__,
            "model_directory": tmp_path / "models",
        }
    )
    valid_model = SimpleNamespace(
        preprocessor=SimpleNamespace(feature_columns=MODEL_FEATURE_COLUMNS)
    )

    _save_artifacts({"valid": valid_model}, config)
    assert (config.model_directory / "valid.joblib").exists()


def test_serving_input_example_excludes_target_and_normalizes_numeric_dtypes():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "amount": [100, 200],
            "transaction_type": ["PAYMENT", "TRANSFER"],
            "is_fraud": [0, 1],
        }
    )

    input_example = _build_serving_input_example(
        frame,
        feature_columns=("amount", "transaction_type"),
    )

    assert list(input_example.columns) == ["amount", "transaction_type"]
    assert input_example["amount"].dtype == "float64"


def test_logged_model_validation_uses_feature_only_input():
    class FeatureOnlyModel:
        def predict_proba(self, frame):
            assert "is_fraud" not in frame.columns
            return [[0.9, 0.1] for _ in range(len(frame))]

        def predict(self, frame):
            assert "is_fraud" not in frame.columns
            return [0 for _ in range(len(frame))]

    class FakeClient:
        def load_model(self, _model_uri):
            return FeatureOnlyModel()

    _validate_logged_model_serving_input(
        FakeClient(),
        SimpleNamespace(model_uri="runs:/run-id/model"),
        _build_serving_input_example(
            pd.DataFrame({"amount": [100], "is_fraud": [1]}),
            feature_columns=("amount",),
        ),
    )
