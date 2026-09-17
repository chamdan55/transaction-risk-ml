"""Training-parameter logging helpers for MLflow runs."""

from __future__ import annotations

from typing import Any

from ml.evaluation.metrics import BinaryClassificationMetrics
from ml.evaluation.threshold import ThresholdEvaluation
from ml.tracking.client import MlflowTrackingClient, TrackingClientError
from ml.tracking.config import TrackingConfig


def log_training_parameters(
    client: MlflowTrackingClient,
    *,
    model_name: str,
    config: TrackingConfig,
    training_config: dict[str, Any],
    dataset_summary: dict[str, Any],
) -> dict[str, Any]:
    """Flatten and log the parameters shared by one training run."""

    if not model_name.strip():
        raise TrackingClientError("model_name must not be empty")
    parameters = {
        "model_name": model_name,
        "experiment_name": config.experiment_name,
        **_flatten_mapping("training", training_config),
        **_flatten_mapping("dataset", dataset_summary),
    }
    parameters = _stringify_parameters(parameters)
    client.log_params(parameters)
    return parameters


def log_validation_metrics(
    client: MlflowTrackingClient,
    *,
    model_name: str,
    selected_evaluation: ThresholdEvaluation,
    threshold_evaluations: list[ThresholdEvaluation] | tuple[ThresholdEvaluation, ...],
) -> dict[str, float]:
    """Log selected validation metrics and the threshold sweep artifact."""

    metrics = _validation_metrics(selected_evaluation.metrics)
    metrics["validation.selected_threshold"] = selected_evaluation.metrics.threshold
    metrics["validation.expected_cost"] = selected_evaluation.expected_cost
    client.log_metrics(metrics)
    client.log_dict(
        {
            "model_name": model_name,
            "evaluations": [evaluation.as_dict() for evaluation in threshold_evaluations],
        },
        f"threshold_analysis/{model_name}.json",
    )
    return metrics


def flatten_parameters(mapping: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    """Flatten nested mappings into MLflow-compatible parameter names."""

    return _flatten_mapping(prefix, mapping) if prefix else _flatten_mapping("", mapping)


def _flatten_mapping(prefix: str, mapping: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in mapping.items():
        parameter_name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_mapping(parameter_name, value))
        else:
            flattened[parameter_name] = value
    return flattened


def _stringify_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value if isinstance(value, (str, int, float, bool)) else str(value)
        for key, value in parameters.items()
    }


def _validation_metrics(metrics: BinaryClassificationMetrics) -> dict[str, float]:
    return {
        "validation.precision": metrics.precision,
        "validation.recall": metrics.recall,
        "validation.f1": metrics.f1,
        "validation.roc_auc": metrics.roc_auc,
        "validation.pr_auc": metrics.pr_auc,
    }
