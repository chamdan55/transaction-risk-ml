import pytest

from ml.evaluation.selection import (
    evaluate_production_candidate_on_test,
    select_production_candidate,
)


def test_candidate_selection_uses_validation_and_final_test_uses_selected_threshold():
    validation_predictions = {
        "logistic_regression": (
            [0, 0, 1, 1],
            [0.60, 0.30, 0.55, 0.70],
        ),
        "random_forest": (
            [0, 0, 1, 1],
            [0.05, 0.80, 0.70, 0.95],
        ),
        "xgboost": (
            [0, 0, 1, 1],
            [0.02, 0.15, 0.85, 0.98],
        ),
    }

    candidate = select_production_candidate(
        validation_predictions,
        thresholds=[0.5, 0.8],
        primary_metric="pr_auc",
        threshold_metric="f1",
    )
    final_result = evaluate_production_candidate_on_test(
        candidate,
        [0, 0, 1, 1],
        [0.10, 0.40, 0.60, 0.90],
    )

    assert candidate.model_name == "xgboost"
    assert candidate.threshold == 0.5
    assert final_result.candidate == candidate
    assert final_result.metrics.threshold == candidate.threshold


def test_candidate_selection_requires_validation_predictions():
    with pytest.raises(ValueError, match="must not be empty"):
        select_production_candidate({}, thresholds=[0.5])
