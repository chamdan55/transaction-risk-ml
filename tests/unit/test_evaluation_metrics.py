import numpy as np
import pytest

from ml.evaluation.metrics import (
    EvaluationError,
    build_calibration_data,
    evaluate_binary_predictions,
)


def test_binary_metrics_include_ranking_and_confusion_matrix_metrics():
    metrics = evaluate_binary_predictions(
        np.array([0, 0, 1, 1]),
        np.array([0.05, 0.40, 0.60, 0.90]),
        threshold=0.5,
    )

    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.roc_auc == 1.0
    assert metrics.pr_auc == 1.0
    assert metrics.brier_score == pytest.approx(0.083125)
    assert metrics.alert_rate == 0.5
    assert (metrics.true_negatives, metrics.false_positives) == (2, 0)
    assert (metrics.false_negatives, metrics.true_positives) == (0, 2)
    assert metrics.as_dict()["threshold"] == 0.5


def test_binary_metrics_change_with_threshold():
    metrics = evaluate_binary_predictions(
        [0, 0, 1, 1],
        [0.05, 0.40, 0.60, 0.90],
        threshold=0.8,
    )

    assert metrics.precision == 1.0
    assert metrics.recall == 0.5
    assert metrics.f1 == pytest.approx(2 / 3)


def test_binary_metrics_reject_invalid_prediction_inputs():
    with pytest.raises(EvaluationError, match="equal length"):
        evaluate_binary_predictions([0, 1], [0.2])
    with pytest.raises(EvaluationError, match="between 0 and 1"):
        evaluate_binary_predictions([0, 1], [-0.1, 0.8])
    with pytest.raises(EvaluationError, match="Both target classes"):
        evaluate_binary_predictions([0, 0], [0.1, 0.2])


def test_calibration_data_is_stable_and_keeps_empty_bins():
    calibration = build_calibration_data([0, 1, 1], [0.1, 0.8, 0.9], n_bins=4)

    assert len(calibration) == 4
    assert calibration[0]["count"] == 1
    assert calibration[3]["count"] == 2
    assert calibration[1]["count"] == 0
