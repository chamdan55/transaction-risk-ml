import pytest

from ml.evaluation.selection import (
    evaluate_production_candidate_on_test,
    evaluate_validation_candidates,
    select_production_candidate,
    select_production_candidate_from_results,
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


def test_candidate_selection_skips_rejected_model_and_preserves_its_diagnostics():
    results = evaluate_validation_candidates(
        {
            "logistic_regression": ([0, 0, 1, 1], [0.10, 0.20, 0.60, 0.70]),
            "xgboost": ([0, 0, 1, 1], [0.05, 0.20, 0.90, 0.95]),
        },
        thresholds=[0.8],
        minimum_recall=0.8,
    )

    candidate = select_production_candidate_from_results(
        results,
        primary_metric="pr_auc",
        threshold_selection_strategy="metric",
        minimum_recall=0.8,
    )

    rejected, eligible = results
    assert candidate.model_name == "xgboost"
    assert not rejected.is_eligible
    assert rejected.rejection_reason is not None
    assert rejected.as_dict()["status"] == "rejected"
    assert rejected.as_dict()["selected_threshold"] is None
    assert len(rejected.as_dict()["threshold_evaluations"]) == 1
    assert eligible.is_eligible


def test_candidate_selection_is_order_independent_for_exact_ties():
    predictions = {
        "zeta": ([0, 0, 1, 1], [0.05, 0.20, 0.90, 0.95]),
        "alpha": ([0, 0, 1, 1], [0.05, 0.20, 0.90, 0.95]),
    }

    forward = select_production_candidate(predictions, thresholds=[0.8])
    reverse = select_production_candidate(dict(reversed(predictions.items())), thresholds=[0.8])

    assert forward.model_name == "alpha"
    assert reverse.model_name == "alpha"


def test_candidate_selection_reports_all_models_when_every_model_is_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "minimum recall.*logistic_regression: maximum_candidate_recall=.*"
            "xgboost: maximum_candidate_recall"
        ),
    ):
        select_production_candidate(
            {
                "xgboost": ([0, 0, 1, 1], [0.05, 0.20, 0.60, 0.70]),
                "logistic_regression": ([0, 0, 1, 1], [0.10, 0.20, 0.55, 0.70]),
            },
            thresholds=[0.8],
            minimum_recall=0.8,
        )


def test_candidate_selection_rejects_invalid_metric_before_quality_gate_evaluation():
    with pytest.raises(ValueError, match="Unsupported model selection metric"):
        select_production_candidate(
            {"xgboost": ([0, 0, 1, 1], [0.05, 0.20, 0.60, 0.70])},
            thresholds=[0.8],
            primary_metric="not_a_metric",
            minimum_recall=0.8,
        )
