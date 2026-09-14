from datetime import datetime, timedelta

import pytest
from pyspark.sql import SparkSession

from ml.data.split import (
    MODEL_FEATURE_COLUMNS,
    chronological_split,
    validate_split_datasets,
    validate_split_ratios,
)


def _transactions(spark: SparkSession):
    start = datetime(2026, 1, 1)
    return spark.createDataFrame(
        [(f"tx-{index:02d}", start + timedelta(hours=index), index % 2) for index in range(10)],
        ["transaction_id", "timestamp", "is_fraud"],
    )


def test_chronological_split_is_deterministic(spark: SparkSession):
    df = _transactions(spark)

    splits = chronological_split(df, 0.7, 0.15, 0.15)

    assert [row["transaction_id"] for row in splits["train"].orderBy("timestamp").collect()] == [
        f"tx-{index:02d}" for index in range(7)
    ]
    assert [row["transaction_id"] for row in splits["validation"].collect()] == ["tx-07"]
    assert [row["transaction_id"] for row in splits["test"].collect()] == ["tx-08", "tx-09"]

    result = validate_split_datasets(df, splits)
    assert result.is_valid
    assert result.total_row_count == 10


def test_split_ratios_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_split_ratios(0.7, 0.2, 0.2)


def test_model_feature_columns_exclude_identifiers_and_target():
    assert "transaction_id" not in MODEL_FEATURE_COLUMNS
    assert "origin_account_id" not in MODEL_FEATURE_COLUMNS
    assert "destination_account_id" not in MODEL_FEATURE_COLUMNS
    assert "timestamp" not in MODEL_FEATURE_COLUMNS
    assert "is_fraud" not in MODEL_FEATURE_COLUMNS
