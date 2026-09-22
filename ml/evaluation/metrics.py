"""Binary classification metrics for Sprint 2 model evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class EvaluationError(ValueError):
    """Raised when predictions cannot be evaluated."""


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    """Evaluation result for one binary classification threshold."""

    threshold: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier_score: float
    alert_rate: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable metrics mapping."""

        return asdict(self)


def evaluate_binary_predictions(
    y_true: Any,
    y_probability: Any,
    *,
    threshold: float = 0.5,
) -> BinaryClassificationMetrics:
    """Calculate ranking and threshold-based binary classification metrics."""

    _validate_threshold(threshold)
    actual = np.asarray(y_true)
    probability = np.asarray(y_probability, dtype=float)
    if actual.ndim != 1 or probability.ndim != 1:
        raise EvaluationError("y_true and y_probability must be one-dimensional")
    if len(actual) != len(probability):
        raise EvaluationError("y_true and y_probability must have equal length")
    if len(actual) == 0:
        raise EvaluationError("Cannot evaluate empty predictions")
    if set(np.unique(actual)).difference({0, 1}):
        raise EvaluationError("y_true must contain only 0 and 1")
    if not np.isfinite(probability).all() or ((probability < 0).any() or (probability > 1).any()):
        raise EvaluationError("y_probability values must be finite and between 0 and 1")
    if len(np.unique(actual)) < 2:
        raise EvaluationError("Both target classes are required for evaluation")

    predicted = (probability >= threshold).astype(int)
    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()
    return BinaryClassificationMetrics(
        threshold=threshold,
        precision=float(precision_score(actual, predicted, zero_division=0)),
        recall=float(recall_score(actual, predicted, zero_division=0)),
        f1=float(f1_score(actual, predicted, zero_division=0)),
        roc_auc=float(roc_auc_score(actual, probability)),
        pr_auc=float(average_precision_score(actual, probability)),
        brier_score=float(brier_score_loss(actual, probability)),
        alert_rate=float(predicted.mean()),
        true_negatives=int(true_negatives),
        false_positives=int(false_positives),
        false_negatives=int(false_negatives),
        true_positives=int(true_positives),
    )


def evaluate_model(
    model: Any,
    frame: Any,
    target: Any,
    *,
    threshold: float = 0.5,
) -> BinaryClassificationMetrics:
    """Evaluate a fitted model exposing ``predict_proba``."""

    try:
        probability = model.predict_proba(frame)[:, 1]
    except (AttributeError, IndexError, TypeError) as exc:
        raise EvaluationError(
            "Model must expose predict_proba with a positive-class probability"
        ) from exc
    return evaluate_binary_predictions(target, probability, threshold=threshold)


def build_calibration_data(
    y_true: Any,
    y_probability: Any,
    *,
    n_bins: int = 10,
) -> list[dict[str, float | int]]:
    """Build deterministic equal-width calibration-bin data.

    Calibration is reported separately from threshold selection so it can be inspected before a
    score is described as a probability. Empty bins are retained with zero counts to keep the
    report schema stable across runs.
    """

    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 2:
        raise EvaluationError("n_bins must be an integer of at least 2")
    actual, probability = _validate_prediction_arrays(y_true, y_probability)
    result: list[dict[str, float | int]] = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for index in range(n_bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == n_bins - 1:
            mask = (probability >= lower) & (probability <= upper)
        else:
            mask = (probability >= lower) & (probability < upper)
        count = int(mask.sum())
        result.append(
            {
                "bin": index,
                "lower_bound": lower,
                "upper_bound": upper,
                "count": count,
                "mean_predicted_probability": float(probability[mask].mean()) if count else 0.0,
                "observed_positive_rate": float(actual[mask].mean()) if count else 0.0,
            }
        )
    return result


# Clear alias for callers that use the report-oriented name.
calibration_report = build_calibration_data


def _validate_prediction_arrays(y_true: Any, y_probability: Any) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true)
    probability = np.asarray(y_probability, dtype=float)
    if actual.ndim != 1 or probability.ndim != 1:
        raise EvaluationError("y_true and y_probability must be one-dimensional")
    if len(actual) != len(probability):
        raise EvaluationError("y_true and y_probability must have equal length")
    if len(actual) == 0:
        raise EvaluationError("Cannot evaluate empty predictions")
    if set(np.unique(actual)).difference({0, 1}):
        raise EvaluationError("y_true must contain only 0 and 1")
    if not np.isfinite(probability).all() or ((probability < 0).any() or (probability > 1).any()):
        raise EvaluationError("y_probability values must be finite and between 0 and 1")
    return actual, probability


def _validate_threshold(threshold: float) -> None:
    if not 0 <= threshold <= 1:
        raise EvaluationError("threshold must be between 0 and 1")
