"""Threshold and business-cost analysis for binary classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from ml.evaluation.metrics import (
    BinaryClassificationMetrics,
    EvaluationError,
    evaluate_binary_predictions,
)

SUPPORTED_THRESHOLD_SELECTION_STRATEGIES = frozenset({"metric", "business_cost"})


@dataclass(frozen=True)
class ThresholdEvaluation:
    """Metrics and expected business cost for one threshold."""

    metrics: BinaryClassificationMetrics
    expected_cost: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable threshold evaluation."""

        result = self.metrics.as_dict()
        result["expected_cost"] = self.expected_cost
        return result


def analyze_thresholds(
    y_true: Any,
    y_probability: Any,
    thresholds: tuple[float, ...] | list[float],
    *,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 1.0,
) -> tuple[ThresholdEvaluation, ...]:
    """Evaluate multiple thresholds against one validation prediction set."""

    _validate_cost("false_positive_cost", false_positive_cost)
    _validate_cost("false_negative_cost", false_negative_cost)
    normalized_thresholds = _normalize_thresholds(thresholds)
    return tuple(
        _evaluate_threshold(
            y_true,
            y_probability,
            threshold=threshold,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        )
        for threshold in normalized_thresholds
    )


def select_best_threshold(
    evaluations: tuple[ThresholdEvaluation, ...] | list[ThresholdEvaluation],
    *,
    metric: str = "f1",
    minimum_recall: float | None = None,
) -> ThresholdEvaluation:
    """Select the validation threshold with the best requested metric."""

    candidates = _filter_by_recall(evaluations, minimum_recall)
    if not candidates:
        raise EvaluationError("No threshold satisfies the minimum recall")
    if not all(hasattr(item.metrics, metric) for item in candidates):
        raise EvaluationError(f"Unsupported threshold selection metric: {metric}")
    return max(
        candidates,
        key=lambda item: (getattr(item.metrics, metric), -item.expected_cost),
    )


def select_business_threshold(
    evaluations: tuple[ThresholdEvaluation, ...] | list[ThresholdEvaluation],
    *,
    minimum_recall: float | None = None,
) -> ThresholdEvaluation:
    """Select the threshold with the lowest expected business cost."""

    candidates = _filter_by_recall(evaluations, minimum_recall)
    if not candidates:
        raise EvaluationError("No threshold satisfies the minimum recall")
    return min(
        candidates,
        key=lambda item: (item.expected_cost, -item.metrics.recall),
    )


def select_threshold(
    evaluations: tuple[ThresholdEvaluation, ...] | list[ThresholdEvaluation],
    *,
    strategy: str = "metric",
    metric: str = "f1",
    minimum_recall: float | None = None,
) -> ThresholdEvaluation:
    """Select a threshold using an explicit metric or business-cost policy."""

    if strategy not in SUPPORTED_THRESHOLD_SELECTION_STRATEGIES:
        raise EvaluationError(
            "Unsupported threshold selection strategy: "
            f"{strategy}; expected one of {sorted(SUPPORTED_THRESHOLD_SELECTION_STRATEGIES)}"
        )
    if strategy == "business_cost":
        return select_business_threshold(evaluations, minimum_recall=minimum_recall)
    return select_best_threshold(
        evaluations,
        metric=metric,
        minimum_recall=minimum_recall,
    )


def _evaluate_threshold(
    y_true: Any,
    y_probability: Any,
    *,
    threshold: float,
    false_positive_cost: float,
    false_negative_cost: float,
) -> ThresholdEvaluation:
    metrics = evaluate_binary_predictions(
        y_true,
        y_probability,
        threshold=threshold,
    )
    expected_cost = (
        metrics.false_positives * false_positive_cost
        + metrics.false_negatives * false_negative_cost
    )
    return ThresholdEvaluation(metrics=metrics, expected_cost=expected_cost)


def _normalize_thresholds(thresholds: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    if not thresholds:
        raise EvaluationError("thresholds must not be empty")
    normalized = tuple(float(threshold) for threshold in thresholds)
    if any(threshold < 0 or threshold > 1 for threshold in normalized):
        raise EvaluationError("thresholds must be between 0 and 1")
    if tuple(sorted(set(normalized))) != normalized:
        raise EvaluationError("thresholds must be sorted and unique")
    return normalized


def _filter_by_recall(
    evaluations: tuple[ThresholdEvaluation, ...] | list[ThresholdEvaluation],
    minimum_recall: float | None,
) -> list[ThresholdEvaluation]:
    if not evaluations:
        raise EvaluationError("evaluations must not be empty")
    if minimum_recall is None:
        return list(evaluations)
    if not 0 <= minimum_recall <= 1:
        raise EvaluationError("minimum_recall must be between 0 and 1")
    return [evaluation for evaluation in evaluations if evaluation.metrics.recall >= minimum_recall]


def _validate_cost(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or not isfinite(float(value)) or value < 0:
        raise EvaluationError(f"{name} must not be negative")
