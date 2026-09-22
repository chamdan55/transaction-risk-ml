"""Typed loading and validation for Sprint 2 model configuration."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from ml.contracts.features import FEATURE_CONTRACT_VERSION

SUPPORTED_PRIMARY_METRICS = frozenset({"precision", "recall", "f1", "roc_auc", "pr_auc"})
REQUIRED_MODELS = frozenset({"logistic_regression", "random_forest", "xgboost"})
SUPPORTED_IMBALANCE_STRATEGIES = frozenset({"negative_sampling", "class_weight", "none"})
SUPPORTED_EVALUATION_SAMPLING_STRATEGIES = frozenset({"full", "exact", "hash"})
SUPPORTED_THRESHOLD_SELECTION_STRATEGIES = frozenset({"metric", "business_cost"})


class ModelConfigurationError(ValueError):
    """Raised when model configuration is missing or invalid."""


@dataclass(frozen=True)
class ModelConfig:
    """Validated configuration used by the Sprint 2 training flow."""

    features_path: Path
    target_column: str
    feature_contract_version: str
    excluded_columns: tuple[str, ...]
    random_seed: int
    primary_metric: str
    imbalance_strategy: str
    evaluation_sampling_strategy: str
    sampling_max_rows: dict[str, int]
    model_params: dict[str, dict[str, Any]]
    thresholds: tuple[float, ...]
    model_directory: Path
    evaluation_report: Path
    model_card: Path = Path("artifacts/model_card.md")
    final_test_sampling_strategy: str = "full"
    false_positive_cost: float = 1.0
    false_negative_cost: float = 1.0
    business_costs_are_assumptions: bool = True
    minimum_recall: float | None = None
    threshold_selection_strategy: str = "metric"
    threshold_metric: str = "f1"
    calibration_bins: int = 10


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
    feature_contract_version = data_config.get(
        "feature_contract_version",
        FEATURE_CONTRACT_VERSION,
    )
    if not isinstance(feature_contract_version, str) or not feature_contract_version.strip():
        raise ModelConfigurationError("feature_contract_version must be a non-empty string")
    if feature_contract_version != FEATURE_CONTRACT_VERSION:
        raise ModelConfigurationError(
            "Unsupported feature_contract_version: "
            f"{feature_contract_version}; expected {FEATURE_CONTRACT_VERSION}"
        )
    excluded_columns = _require_string_tuple(data_config, "excluded_columns")
    random_seed = _require_int(training_config, "random_seed", minimum=0)
    primary_metric = _require_non_empty_string(training_config, "primary_metric")
    imbalance_strategy = _require_non_empty_string(
        {"imbalance_strategy": training_config.get("imbalance_strategy", "negative_sampling")},
        "imbalance_strategy",
    )
    # ``balanced`` was the pre-TRM-002 name for model-side class weighting. Keep it readable for
    # old local configs while making the selected strategy explicit in new configurations.
    if imbalance_strategy == "balanced":
        imbalance_strategy = "class_weight"
    if imbalance_strategy not in SUPPORTED_IMBALANCE_STRATEGIES:
        raise ModelConfigurationError(
            f"Unsupported imbalance_strategy: {imbalance_strategy}. "
            f"Expected one of {sorted(SUPPORTED_IMBALANCE_STRATEGIES)}"
        )
    sampling_max_rows = _require_sampling_max_rows(training_config)
    evaluation_sampling_strategy = _require_non_empty_string(
        {
            "evaluation_sampling_strategy": training_config.get(
                "evaluation_sampling_strategy", "full"
            )
        },
        "evaluation_sampling_strategy",
    )
    if evaluation_sampling_strategy not in SUPPORTED_EVALUATION_SAMPLING_STRATEGIES:
        raise ModelConfigurationError(
            f"Unsupported evaluation_sampling_strategy: {evaluation_sampling_strategy}. "
            f"Expected one of {sorted(SUPPORTED_EVALUATION_SAMPLING_STRATEGIES)}"
        )
    final_test_sampling_strategy = _require_sampling_strategy(
        training_config.get(
            "final_test_sampling_strategy",
            evaluation_config.get("final_test_sampling_strategy", "full"),
        ),
        key="final_test_sampling_strategy",
    )
    false_positive_cost, false_negative_cost, business_costs_are_assumptions = (
        _require_business_costs(evaluation_config)
    )
    minimum_recall = _require_optional_ratio(evaluation_config, "minimum_recall")
    threshold_selection_strategy = _require_non_empty_string(
        {
            "threshold_selection_strategy": evaluation_config.get(
                "threshold_selection_strategy", "metric"
            )
        },
        "threshold_selection_strategy",
    )
    if threshold_selection_strategy not in SUPPORTED_THRESHOLD_SELECTION_STRATEGIES:
        raise ModelConfigurationError(
            f"Unsupported threshold_selection_strategy: {threshold_selection_strategy}. "
            f"Expected one of {sorted(SUPPORTED_THRESHOLD_SELECTION_STRATEGIES)}"
        )
    threshold_metric = _require_non_empty_string(
        {"threshold_metric": evaluation_config.get("threshold_metric", "f1")},
        "threshold_metric",
    )
    if threshold_metric not in SUPPORTED_PRIMARY_METRICS:
        raise ModelConfigurationError(
            f"Unsupported threshold_metric: {threshold_metric}. "
            f"Expected one of {sorted(SUPPORTED_PRIMARY_METRICS)}"
        )
    calibration_bins = _require_int(
        {"calibration_bins": evaluation_config.get("calibration_bins", 10)},
        "calibration_bins",
        minimum=2,
    )
    thresholds = _require_thresholds(evaluation_config)
    model_directory = Path(outputs_config.get("model_directory", "artifacts/models"))
    evaluation_report = Path(
        outputs_config.get("evaluation_report", "artifacts/evaluation_report.json")
    )
    model_card = Path(outputs_config.get("model_card", "artifacts/model_card.md"))

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
    _validate_imbalance_model_params(imbalance_strategy, model_params)

    return ModelConfig(
        features_path=Path(features_path),
        target_column=target_column,
        feature_contract_version=feature_contract_version,
        excluded_columns=excluded_columns,
        random_seed=random_seed,
        primary_metric=primary_metric,
        imbalance_strategy=imbalance_strategy,
        evaluation_sampling_strategy=evaluation_sampling_strategy,
        sampling_max_rows=sampling_max_rows,
        model_params=model_params,
        thresholds=thresholds,
        model_directory=model_directory,
        evaluation_report=evaluation_report,
        model_card=model_card,
        final_test_sampling_strategy=final_test_sampling_strategy,
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
        business_costs_are_assumptions=business_costs_are_assumptions,
        minimum_recall=minimum_recall,
        threshold_selection_strategy=threshold_selection_strategy,
        threshold_metric=threshold_metric,
        calibration_bins=calibration_bins,
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


def _require_sampling_strategy(value: Any, *, key: str) -> str:
    if not isinstance(value, str) or value not in SUPPORTED_EVALUATION_SAMPLING_STRATEGIES:
        raise ModelConfigurationError(
            f"{key} must be one of {sorted(SUPPORTED_EVALUATION_SAMPLING_STRATEGIES)}"
        )
    return value


def _require_nonnegative_float(mapping: dict[str, Any], key: str, *, default: float) -> float:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelConfigurationError(f"{key} must be a non-negative number")
    value = float(value)
    if value < 0 or value != value or value in (float("inf"), float("-inf")):
        raise ModelConfigurationError(f"{key} must be a non-negative finite number")
    return value


def _require_optional_ratio(mapping: dict[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelConfigurationError(f"{key} must be between 0 and 1")
    value = float(value)
    if not isfinite(value) or value < 0 or value > 1:
        raise ModelConfigurationError(f"{key} must be between 0 and 1")
    return value


def _require_business_costs(evaluation_config: dict[str, Any]) -> tuple[float, float, bool]:
    nested = evaluation_config.get("business_costs", {})
    if nested is None:
        nested = {}
    if not isinstance(nested, dict):
        raise ModelConfigurationError("business_costs must be a mapping")
    false_positive_cost = _require_nonnegative_float(
        evaluation_config,
        "false_positive_cost",
        default=nested.get("false_positive", 1.0),
    )
    false_negative_cost = _require_nonnegative_float(
        evaluation_config,
        "false_negative_cost",
        default=nested.get("false_negative", 1.0),
    )
    assumptions = evaluation_config.get(
        "business_costs_are_assumptions",
        nested.get("assumptions", True),
    )
    if not isinstance(assumptions, bool):
        raise ModelConfigurationError("business_costs_are_assumptions must be a boolean")
    return false_positive_cost, false_negative_cost, assumptions


def _validate_imbalance_model_params(
    imbalance_strategy: str,
    model_params: dict[str, dict[str, Any]],
) -> None:
    """Reject accidental double weighting when negative sampling is selected."""

    if imbalance_strategy != "negative_sampling":
        return
    for model_name in ("logistic_regression", "random_forest"):
        if model_params[model_name].get("class_weight") == "balanced":
            raise ModelConfigurationError(
                f"{model_name}.class_weight=balanced conflicts with negative_sampling"
            )
    scale_pos_weight = model_params["xgboost"].get("scale_pos_weight")
    if scale_pos_weight not in (None, 1, 1.0):
        raise ModelConfigurationError(
            "xgboost.scale_pos_weight conflicts with negative_sampling; omit it or set 1.0"
        )


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
