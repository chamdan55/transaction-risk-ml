"""Production-candidate selection and final test evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ml.evaluation.metrics import (
    BinaryClassificationMetrics,
    EvaluationError,
    evaluate_binary_predictions,
)
from ml.evaluation.threshold import (
    ThresholdEvaluation,
    analyze_thresholds,
    select_best_threshold,
)


@dataclass(frozen=True)
class CandidateValidationResult:
    """Validation result used to compare one model candidate."""

    model_name: str
    selected_threshold: ThresholdEvaluation
    selection_metric: str

    @property
    def selection_score(self) -> float:
        """Return the candidate score used for model selection."""

        return float(getattr(self.selected_threshold.metrics, self.selection_metric))


@dataclass(frozen=True)
class ProductionCandidate:
    """Selected model and threshold, based only on validation data."""

    model_name: str
    threshold: float
    selection_metric: str
    validation_metrics: BinaryClassificationMetrics
    validation_score: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable candidate summary."""

        return {
            "model_name": self.model_name,
            "threshold": self.threshold,
            "selection_metric": self.selection_metric,
            "validation_score": self.validation_score,
            "validation_metrics": self.validation_metrics.as_dict(),
        }


@dataclass(frozen=True)
class FinalTestResult:
    """Final test evaluation tied to a selected production candidate."""

    candidate: ProductionCandidate
    metrics: BinaryClassificationMetrics

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable final evaluation report."""

        return {
            "candidate": self.candidate.as_dict(),
            "test_metrics": self.metrics.as_dict(),
        }


def select_production_candidate(
    validation_predictions: dict[str, tuple[Any, Any]],
    *,
    thresholds: tuple[float, ...] | list[float],
    primary_metric: str = "pr_auc",
    threshold_metric: str = "f1",
    minimum_recall: float | None = None,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 1.0,
) -> ProductionCandidate:
    """Select a model and threshold using validation predictions only."""

    if not validation_predictions:
        raise EvaluationError("validation_predictions must not be empty")
    candidate_results = []
    for model_name, (y_true, y_probability) in validation_predictions.items():
        evaluations = analyze_thresholds(
            y_true,
            y_probability,
            thresholds,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        )
        selected_threshold = select_best_threshold(
            evaluations,
            metric=threshold_metric,
            minimum_recall=minimum_recall,
        )
        if not hasattr(selected_threshold.metrics, primary_metric):
            raise EvaluationError(f"Unsupported model selection metric: {primary_metric}")
        candidate_results.append(
            CandidateValidationResult(
                model_name=model_name,
                selected_threshold=selected_threshold,
                selection_metric=primary_metric,
            )
        )

    selected = max(
        candidate_results,
        key=lambda result: (result.selection_score, -result.selected_threshold.expected_cost),
    )
    return ProductionCandidate(
        model_name=selected.model_name,
        threshold=selected.selected_threshold.metrics.threshold,
        selection_metric=primary_metric,
        validation_metrics=selected.selected_threshold.metrics,
        validation_score=selected.selection_score,
    )


def evaluate_production_candidate_on_test(
    candidate: ProductionCandidate,
    y_true: Any,
    y_probability: Any,
) -> FinalTestResult:
    """Run the final test evaluation with the already-selected threshold."""

    test_metrics = evaluate_binary_predictions(
        y_true,
        y_probability,
        threshold=candidate.threshold,
    )
    return FinalTestResult(candidate=candidate, metrics=test_metrics)
