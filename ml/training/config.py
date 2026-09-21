"""Typed loading and validation for Sprint 2 model configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_PRIMARY_METRICS = frozenset({"precision", "recall", "f1", "roc_auc", "pr_auc"})
REQUIRED_MODELS = frozenset({"logistic_regression", "random_forest", "xgboost"})


class ModelConfigurationError(ValueError):
    """Raised when model configuration is missing or invalid."""


@dataclass(frozen=True)
class ModelConfig:
    """Validated configuration used by the Sprint 2 training flow."""

    features_path: Path
    target_column: str
    excluded_columns: tuple[str, ...]
    random_seed: int
    primary_metric: str
    imbalance_strategy: str
    sampling_max_rows: dict[str, int]
    model_params: dict[str, dict[str, Any]]
    thresholds: tuple[float, ...]
    model_directory: Path
    evaluation_report: Path


def load_model_config(config_path: str | Path) -> ModelConfig:
    """Load and validate a model configuration YAML file."""

    path = Path(config_path)
    try:
        with path.open(encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)
    except OSError as exc:
        raise ModelConfigurationError(f"Unable to read model configuration from {path}") from exc
    except yaml.YAMLError as exc:
        raise ModelConfigurationError(f"Invalid YAML in model configuration: {path}") from exc

    if not isinstance(raw_config, dict):
        raise ModelConfigurationError("Model configuration must be a mapping")

    data_config = _require_mapping(raw_config, "data")
    training_config = _require_mapping(raw_config, "training")
    models_config = _require_mapping(raw_config, "models")
    evaluation_config = _require_mapping(raw_config, "evaluation")
    outputs_config = raw_config.get("outputs", {})
    if not isinstance(outputs_config, dict):
        raise ModelConfigurationError("outputs must be a mapping")

    features_path = _require_non_empty_string(data_config, "features_path")
    target_column = _require_non_empty_string(data_config, "target_column")
    excluded_columns = _require_string_tuple(data_config, "excluded_columns")
    random_seed = _require_int(training_config, "random_seed", minimum=0)
    primary_metric = _require_non_empty_string(training_config, "primary_metric")
    imbalance_strategy = _require_non_empty_string(
        {"imbalance_strategy": training_config.get("imbalance_strategy", "balanced")},
        "imbalance_strategy",
    )
    sampling_max_rows = _require_sampling_max_rows(training_config)
    thresholds = _require_thresholds(evaluation_config)
    model_directory = Path(outputs_config.get("model_directory", "artifacts/models"))
    evaluation_report = Path(
        outputs_config.get("evaluation_report", "artifacts/evaluation_report.json")
    )

    if primary_metric not in SUPPORTED_PRIMARY_METRICS:
        raise ModelConfigurationError(
            f"Unsupported primary_metric: {primary_metric}. "
            f"Expected one of {sorted(SUPPORTED_PRIMARY_METRICS)}"
        )
    if target_column not in excluded_columns:
        raise ModelConfigurationError("target_column must be included in excluded_columns")

    missing_models = REQUIRED_MODELS.difference(models_config)
    if missing_models:
        raise ModelConfigurationError(f"Missing model configuration: {sorted(missing_models)}")
    model_params = {
        model_name: _require_mapping(models_config, model_name) for model_name in REQUIRED_MODELS
    }

    return ModelConfig(
        features_path=Path(features_path),
        target_column=target_column,
        excluded_columns=excluded_columns,
        random_seed=random_seed,
        primary_metric=primary_metric,
        imbalance_strategy=imbalance_strategy,
        sampling_max_rows=sampling_max_rows,
        model_params=model_params,
        thresholds=thresholds,
        model_directory=model_directory,
        evaluation_report=evaluation_report,
    )


def _require_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ModelConfigurationError(f"{key} must be a mapping")
    return value


def _require_non_empty_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ModelConfigurationError(f"{key} must be a non-empty string")
    return value


def _require_string_tuple(mapping: dict[str, Any], key: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ModelConfigurationError(f"{key} must be a list of non-empty strings")
    if len(set(value)) != len(value):
        raise ModelConfigurationError(f"{key} must not contain duplicates")
    return tuple(value)


def _require_int(mapping: dict[str, Any], key: str, *, minimum: int | None = None) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelConfigurationError(f"{key} must be an integer")
    if minimum is not None and value < minimum:
        raise ModelConfigurationError(f"{key} must be at least {minimum}")
    return value


def _require_sampling_max_rows(training_config: dict[str, Any]) -> dict[str, int]:
    sampling_config = _require_mapping(training_config, "sampling")
    max_rows = _require_mapping(sampling_config, "max_rows")
    expected_splits = {"train", "validation", "test"}
    if set(max_rows) != expected_splits:
        raise ModelConfigurationError(
            f"sampling.max_rows must contain exactly: {sorted(expected_splits)}"
        )
    return {
        split_name: _require_int(max_rows, split_name, minimum=1)
        for split_name in sorted(expected_splits)
    }


def _require_thresholds(mapping: dict[str, Any]) -> tuple[float, ...]:
    value = mapping.get("thresholds")
    if not isinstance(value, list) or not value:
        raise ModelConfigurationError("thresholds must be a non-empty list")
    try:
        thresholds = tuple(float(threshold) for threshold in value)
    except (TypeError, ValueError) as exc:
        raise ModelConfigurationError("thresholds must contain numbers") from exc
    if any(threshold <= 0 or threshold >= 1 for threshold in thresholds):
        raise ModelConfigurationError("thresholds must be greater than 0 and less than 1")
    if tuple(sorted(set(thresholds))) != thresholds:
        raise ModelConfigurationError("thresholds must be sorted and unique")
    return thresholds
