import pytest

from ml.evaluation.threshold import (
    analyze_thresholds,
    select_best_threshold,
    select_business_threshold,
)


def test_threshold_analysis_calculates_business_costs():
    evaluations = analyze_thresholds(
        [0, 0, 1, 1],
        [0.05, 0.40, 0.60, 0.90],
        [0.5, 0.8],
        false_positive_cost=1,
        false_negative_cost=5,
    )

    assert [item.metrics.threshold for item in evaluations] == [0.5, 0.8]
    assert [item.expected_cost for item in evaluations] == [0.0, 5.0]
    assert select_business_threshold(evaluations).metrics.threshold == 0.5


def test_threshold_selection_can_apply_minimum_recall_constraint():
    evaluations = analyze_thresholds(
        [0, 0, 1, 1],
        [0.05, 0.40, 0.60, 0.90],
        [0.5, 0.8],
    )

    selected = select_best_threshold(evaluations, metric="precision", minimum_recall=0.75)
    assert selected.metrics.threshold == 0.5


def test_threshold_analysis_rejects_invalid_thresholds_and_costs():
    with pytest.raises(ValueError, match="sorted and unique"):
        analyze_thresholds([0, 1], [0.2, 0.8], [0.8, 0.2])
    with pytest.raises(ValueError, match="must not be negative"):
        analyze_thresholds([0, 1], [0.2, 0.8], [0.5], false_positive_cost=-1)
