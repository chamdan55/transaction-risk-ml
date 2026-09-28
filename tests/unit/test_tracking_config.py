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
    assert config.force_proxy_artifact_transfers is False


def test_tracking_config_rejects_missing_required_value(monkeypatch):
    invalid_config = """
tracking:
  uri: mlruns
  experiment_name: transaction-risk-classification
"""
    monkeypatch.setattr(
        Path,
        "open",
        lambda *_args, **_kwargs: StringIO(invalid_config),
    )

    with pytest.raises(TrackingConfigurationError, match="registered_model_name"):
        load_tracking_config("virtual-tracking.yaml")


def test_container_tracking_config_uses_server_managed_artifact_location():
    config = load_tracking_config(PROJECT_ROOT / "configs" / "tracking-compose.yaml")

    assert config.uri == "http://127.0.0.1:5000"
    assert config.experiment_name == "transaction-risk-classification-container"
    assert config.artifact_location is None
    assert config.force_proxy_artifact_transfers is True


def test_tracking_config_rejects_nonboolean_proxy_setting(monkeypatch):
    invalid_config = """
tracking:
  uri: http://127.0.0.1:5000
  experiment_name: remote-experiment
  registered_model_name: risk-model
  force_proxy_artifact_transfers: "true"
"""
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: StringIO(invalid_config))

    with pytest.raises(TrackingConfigurationError, match="force_proxy_artifact_transfers"):
        load_tracking_config("virtual-tracking.yaml")


def test_tracking_uri_override_supports_non_default_compose_port():
    config = load_tracking_config(
        PROJECT_ROOT / "configs" / "tracking-compose.yaml",
        tracking_uri="http://127.0.0.1:5055",
    )

    assert config.uri == "http://127.0.0.1:5055"
    assert config.artifact_location is None


def test_tracking_uri_override_rejects_empty_value():
    with pytest.raises(TrackingConfigurationError, match="tracking_uri override"):
        load_tracking_config(TRACKING_CONFIG_PATH, tracking_uri=" ")
