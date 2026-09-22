"""Production-candidate selection and final test evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from ml.evaluation.metrics import (
    BinaryClassificationMetrics,
    EvaluationError,
    evaluate_binary_predictions,
)
from ml.evaluation.threshold import (
    SUPPORTED_THRESHOLD_SELECTION_STRATEGIES,
    ThresholdEvaluation,
    analyze_thresholds,
    select_threshold,
)


@dataclass(frozen=True)
class CandidateValidationResult:
    """Complete validation outcome for one model, including quality-gate rejection."""

    model_name: str
    threshold_evaluations: tuple[ThresholdEvaluation, ...]
    selection_metric: str
    selected_threshold: ThresholdEvaluation | None = None
    rejection_reason: str | None = None

    @property
    def is_eligible(self) -> bool:
        """Whether this model has a threshold that satisfies the quality constraints."""

        return self.selected_threshold is not None

    @property
    def maximum_candidate_recall(self) -> float:
        """Return the highest recall observed across configured thresholds."""

        return max(item.metrics.recall for item in self.threshold_evaluations)

    @property
    def selection_score(self) -> float:
        """Return the candidate score used for model selection."""

        if self.selected_threshold is None:
            raise EvaluationError(f"Model {self.model_name} is not eligible for selection")
        return float(getattr(self.selected_threshold.metrics, self.selection_metric))

    def as_dict(self) -> dict[str, Any]:
        """Return the complete, JSON-serializable validation decision."""

        selected = self.selected_threshold
        return {
            "model_name": self.model_name,
            "status": "eligible" if self.is_eligible else "rejected",
            "selection_metric": self.selection_metric,
            "maximum_candidate_recall": self.maximum_candidate_recall,
            "rejection_reason": self.rejection_reason,
            "selected_threshold": None if selected is None else selected.metrics.threshold,
            "metrics": None if selected is None else selected.metrics.as_dict(),
            "expected_cost": None if selected is None else selected.expected_cost,
            "threshold_evaluations": [
                evaluation.as_dict() for evaluation in self.threshold_evaluations
            ],
        }


@dataclass(frozen=True)
class ProductionCandidate:
    """Selected model and threshold, based only on validation data."""

    model_name: str
    threshold: float
    selection_metric: str
    validation_metrics: BinaryClassificationMetrics
    validation_score: float
    threshold_selection_strategy: str = "metric"
    expected_cost: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable candidate summary."""

        return {
            "model_name": self.model_name,
            "threshold": self.threshold,
            "selection_metric": self.selection_metric,
            "validation_score": self.validation_score,
            "threshold_selection_strategy": self.threshold_selection_strategy,
            "expected_cost": self.expected_cost,
            "validation_metrics": self.validation_metrics.as_dict(),
        }


@dataclass(frozen=True)
class FinalTestResult:
    """Final test evaluation tied to a selected production candidate."""

    candidate: ProductionCandidate
    metrics: BinaryClassificationMetrics
    expected_cost: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable final evaluation report."""

        return {
            "candidate": self.candidate.as_dict(),
            "test_metrics": self.metrics.as_dict(),
            "expected_cost": self.expected_cost,
        }


def select_production_candidate(
    validation_predictions: dict[str, tuple[Any, Any]],
    *,
    thresholds: tuple[float, ...] | list[float],
    primary_metric: str = "pr_auc",
    threshold_metric: str = "f1",
    threshold_selection_strategy: str = "metric",
    minimum_recall: float | None = None,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 1.0,
) -> ProductionCandidate:
    """Select a model and threshold using validation predictions only."""

    results = evaluate_validation_candidates(
        validation_predictions,
        thresholds=thresholds,
        primary_metric=primary_metric,
        threshold_metric=threshold_metric,
        threshold_selection_strategy=threshold_selection_strategy,
        minimum_recall=minimum_recall,
        false_positive_cost=false_positive_cost,
        false_negative_cost=false_negative_cost,
    )
    return select_production_candidate_from_results(
        results,
        primary_metric=primary_metric,
        threshold_selection_strategy=threshold_selection_strategy,
        minimum_recall=minimum_recall,
    )


def evaluate_validation_candidates(
    validation_predictions: dict[str, tuple[Any, Any]],
    *,
    thresholds: tuple[float, ...] | list[float],
    primary_metric: str = "pr_auc",
    threshold_metric: str = "f1",
    threshold_selection_strategy: str = "metric",
    minimum_recall: float | None = None,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 1.0,
) -> tuple[CandidateValidationResult, ...]:
    """Evaluate every model without letting one quality-gate rejection abort the set."""

    if not validation_predictions:
        raise EvaluationError("validation_predictions must not be empty")
    _validate_selection_configuration(
        primary_metric=primary_metric,
        threshold_metric=threshold_metric,
        threshold_selection_strategy=threshold_selection_strategy,
        minimum_recall=minimum_recall,
    )
    candidate_results: list[CandidateValidationResult] = []
    for model_name, (y_true, y_probability) in validation_predictions.items():
        evaluations = analyze_thresholds(
            y_true,
            y_probability,
            thresholds,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        )
        eligible_evaluations = _eligible_evaluations(evaluations, minimum_recall)
        if not eligible_evaluations:
            candidate_results.append(
                CandidateValidationResult(
                    model_name=model_name,
                    threshold_evaluations=evaluations,
                    selection_metric=primary_metric,
                    rejection_reason=(
                        "No threshold satisfies the minimum recall "
                        f"({minimum_recall:.6f}); maximum candidate recall="
                        f"{max(item.metrics.recall for item in evaluations):.6f}"
                    ),
                )
            )
            continue
        selected_threshold = select_threshold(
            evaluations,
            strategy=threshold_selection_strategy,
            metric=threshold_metric,
            minimum_recall=minimum_recall,
        )
        candidate_results.append(
            CandidateValidationResult(
                model_name=model_name,
                threshold_evaluations=evaluations,
                selected_threshold=selected_threshold,
                selection_metric=primary_metric,
            )
        )
    return tuple(candidate_results)


def select_production_candidate_from_results(
    candidate_results: tuple[CandidateValidationResult, ...] | list[CandidateValidationResult],
    *,
    primary_metric: str,
    threshold_selection_strategy: str,
    minimum_recall: float | None,
) -> ProductionCandidate:
    """Rank eligible validation results and fail only when every model is rejected."""

    if not candidate_results:
        raise EvaluationError("candidate_results must not be empty")
    if any(result.selection_metric != primary_metric for result in candidate_results):
        raise EvaluationError("candidate result selection metric does not match primary_metric")
    eligible_results = [result for result in candidate_results if result.is_eligible]
    if not eligible_results:
        details = "; ".join(
            f"{result.model_name}: maximum_candidate_recall={result.maximum_candidate_recall:.6f}"
            for result in sorted(candidate_results, key=lambda item: item.model_name)
        )
        raise EvaluationError(
            "No model has a threshold satisfying the minimum recall "
            f"({minimum_recall!r}). {details}"
        )

    selected = sorted(
        eligible_results,
        key=lambda result: (
            -result.selection_score,
            result.selected_threshold.expected_cost if result.selected_threshold else float("inf"),
            result.model_name,
        ),
    )[0]
    if selected.selected_threshold is None:  # pragma: no cover - guarded by is_eligible above
        raise EvaluationError(f"Model {selected.model_name} is not eligible for selection")
    return ProductionCandidate(
        model_name=selected.model_name,
        threshold=selected.selected_threshold.metrics.threshold,
        selection_metric=primary_metric,
        validation_metrics=selected.selected_threshold.metrics,
        validation_score=selected.selection_score,
        threshold_selection_strategy=threshold_selection_strategy,
        expected_cost=selected.selected_threshold.expected_cost,
    )


def _eligible_evaluations(
    evaluations: tuple[ThresholdEvaluation, ...], minimum_recall: float | None
) -> tuple[ThresholdEvaluation, ...]:
    if minimum_recall is None:
        return evaluations
    return tuple(
        evaluation for evaluation in evaluations if evaluation.metrics.recall >= minimum_recall
    )


def _validate_selection_configuration(
    *,
    primary_metric: str,
    threshold_metric: str,
    threshold_selection_strategy: str,
    minimum_recall: float | None,
) -> None:
    available_metrics = BinaryClassificationMetrics.__dataclass_fields__
    if primary_metric not in available_metrics:
        raise EvaluationError(f"Unsupported model selection metric: {primary_metric}")
    if threshold_selection_strategy not in SUPPORTED_THRESHOLD_SELECTION_STRATEGIES:
        raise EvaluationError(
            "Unsupported threshold selection strategy: "
            f"{threshold_selection_strategy}; expected one of "
            f"{sorted(SUPPORTED_THRESHOLD_SELECTION_STRATEGIES)}"
        )
    if threshold_selection_strategy == "metric" and threshold_metric not in available_metrics:
        raise EvaluationError(f"Unsupported threshold selection metric: {threshold_metric}")
    if minimum_recall is not None and not 0 <= minimum_recall <= 1:
        raise EvaluationError("minimum_recall must be between 0 and 1")


def evaluate_production_candidate_on_test(
    candidate: ProductionCandidate,
    y_true: Any,
    y_probability: Any,
    *,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 1.0,
) -> FinalTestResult:
    """Run the final test evaluation with the already-selected threshold."""

    if (
        not isfinite(float(false_positive_cost))
        or not isfinite(float(false_negative_cost))
        or false_positive_cost < 0
        or false_negative_cost < 0
    ):
        raise EvaluationError("Business costs must be finite and non-negative")
    test_metrics = evaluate_binary_predictions(
        y_true,
        y_probability,
        threshold=candidate.threshold,
    )
    expected_cost = (
        test_metrics.false_positives * false_positive_cost
        + test_metrics.false_negatives * false_negative_cost
    )
    return FinalTestResult(
        candidate=candidate,
        metrics=test_metrics,
        expected_cost=float(expected_cost),
    )
