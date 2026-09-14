from pathlib import Path

import pytest

from ml.training.config import ModelConfigurationError, load_model_config

PROJECT_ROOT = Path(__file__).parents[2]
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"


def test_model_config_loads_repository_configuration():
    config = load_model_config(MODEL_CONFIG_PATH)

    assert config.features_path == Path("data/processed/features")
    assert config.target_column == "is_fraud"
    assert config.random_seed == 42
    assert config.primary_metric == "pr_auc"
    assert config.thresholds == (0.1, 0.2, 0.3, 0.4, 0.5)
    assert {"logistic_regression", "random_forest", "xgboost"} == set(config.model_params)


def test_model_config_rejects_missing_model(tmp_path):
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        """
data:
  features_path: data/processed/features
  target_column: is_fraud
  excluded_columns: [is_fraud]
training:
  random_seed: 42
  primary_metric: pr_auc
models:
  logistic_regression: {}
  random_forest: {}
evaluation:
  thresholds: [0.5]
""",
        encoding="utf-8",
    )

    with pytest.raises(ModelConfigurationError, match="Missing model configuration"):
        load_model_config(config_path)


def test_model_config_rejects_unsorted_thresholds(tmp_path):
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        """
data:
  features_path: data/processed/features
  target_column: is_fraud
  excluded_columns: [is_fraud]
training:
  random_seed: 42
  primary_metric: pr_auc
models:
  logistic_regression: {}
  random_forest: {}
  xgboost: {}
evaluation:
  thresholds: [0.5, 0.1]
""",
        encoding="utf-8",
    )

    with pytest.raises(ModelConfigurationError, match="sorted and unique"):
        load_model_config(config_path)
