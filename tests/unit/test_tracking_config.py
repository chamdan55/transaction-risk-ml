from io import StringIO
from pathlib import Path

import pytest

from ml.tracking.config import TrackingConfigurationError, load_tracking_config

PROJECT_ROOT = Path(__file__).parents[2]
TRACKING_CONFIG_PATH = PROJECT_ROOT / "configs" / "tracking.yaml"


def test_tracking_config_loads_repository_configuration():
    config = load_tracking_config(TRACKING_CONFIG_PATH)

    assert config.uri == "sqlite:///mlflow.db"
    assert config.experiment_name == "transaction-risk-classification"
    assert config.registered_model_name == "transaction-risk-model"
    assert config.artifact_location == "mlartifacts"


def test_tracking_config_rejects_missing_required_value(monkeypatch):
    invalid_config = """
tracking:
  uri: mlruns
  experiment_name: transaction-risk-classification
  registered_model_name: transaction-risk-model
"""
    monkeypatch.setattr(
        Path,
        "open",
        lambda *_args, **_kwargs: StringIO(invalid_config),
    )

    with pytest.raises(TrackingConfigurationError, match="artifact_location"):
        load_tracking_config("virtual-tracking.yaml")
