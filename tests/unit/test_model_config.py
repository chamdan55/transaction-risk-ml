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
    assert config.imbalance_strategy == "negative_sampling"
    assert config.evaluation_sampling_strategy == "full"
    assert config.final_test_sampling_strategy == "full"
    assert config.false_positive_cost == 1.0
    assert config.false_negative_cost == 5.0
    assert config.business_costs_are_assumptions is True
    assert config.minimum_recall == 0.8
    assert config.threshold_selection_strategy == "business_cost"
    assert config.calibration_bins == 10
    assert config.thresholds == (0.1, 0.2, 0.3, 0.4, 0.5)
    assert config.sampling_max_rows == {"test": 100000, "train": 200000, "validation": 100000}
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
  sampling:
    max_rows: {train: 10, validation: 10, test: 10}
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
  sampling:
    max_rows: {train: 10, validation: 10, test: 10}
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


def test_model_config_rejects_unknown_feature_contract(tmp_path):
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        """
data:
  features_path: data/processed/features
  target_column: is_fraud
  feature_contract_version: post-event-v1
  excluded_columns: [is_fraud]
training:
  random_seed: 42
  primary_metric: pr_auc
  sampling:
    max_rows: {train: 10, validation: 10, test: 10}
models:
  logistic_regression: {}
  random_forest: {}
  xgboost: {}
evaluation:
  thresholds: [0.5]
""",
        encoding="utf-8",
    )

    with pytest.raises(ModelConfigurationError, match="Unsupported feature_contract_version"):
        load_model_config(config_path)


def test_model_config_rejects_double_imbalance_handling(tmp_path):
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
  imbalance_strategy: negative_sampling
  sampling:
    max_rows: {train: 10, validation: 10, test: 10}
models:
  logistic_regression:
    class_weight: balanced
  random_forest: {}
  xgboost: {}
evaluation:
  thresholds: [0.5]
""",
        encoding="utf-8",
    )

    with pytest.raises(ModelConfigurationError, match="conflicts with negative_sampling"):
        load_model_config(config_path)


def test_model_config_rejects_invalid_business_cost_and_recall(tmp_path):
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
  sampling:
    max_rows: {train: 10, validation: 10, test: 10}
models:
  logistic_regression: {}
  random_forest: {}
  xgboost: {}
evaluation:
  false_negative_cost: -1
  minimum_recall: 1.5
  thresholds: [0.5]
""",
        encoding="utf-8",
    )

    with pytest.raises(ModelConfigurationError, match="false_negative_cost"):
        load_model_config(config_path)
