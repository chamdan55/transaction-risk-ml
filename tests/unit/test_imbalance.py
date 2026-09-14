import pandas as pd
import pytest

from ml.training.imbalance import (
    ClassImbalanceError,
    balanced_class_weight,
    scale_pos_weight,
    summarize_target,
)


def test_target_distribution_and_imbalance_weights():
    distribution = summarize_target(pd.Series([0, 0, 0, 1]))

    assert distribution.total_count == 4
    assert distribution.positive_count == 1
    assert distribution.negative_count == 3
    assert distribution.positive_rate == 0.25
    assert balanced_class_weight(distribution) == {0: 2 / 3, 1: 2.0}
    assert scale_pos_weight(distribution) == 3.0


def test_target_distribution_rejects_invalid_values():
    with pytest.raises(ClassImbalanceError, match="only 0 and 1"):
        summarize_target(pd.Series([0, 1, 2]))


def test_target_distribution_rejects_null_values():
    with pytest.raises(ClassImbalanceError, match="null values"):
        summarize_target(pd.Series([0, 1, None]))


def test_class_weights_require_both_classes():
    distribution = summarize_target(pd.Series([0, 0]))

    with pytest.raises(ClassImbalanceError, match="Both positive and negative"):
        balanced_class_weight(distribution)
