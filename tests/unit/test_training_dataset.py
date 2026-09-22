from __future__ import annotations

import pytest
from pyspark.sql import SparkSession

from ml.training.dataset import (
    TrainingDatasetContractError,
    load_training_dataset,
)

MODEL_FEATURES = ("amount_log", "transactions_last_1h")


def _write_splits(spark: SparkSession, root, rows_by_split: dict[str, list[tuple]]):
    for split_name, rows in rows_by_split.items():
        columns = [
            "transaction_id",
            "amount_log",
            "transactions_last_1h",
            "is_fraud",
        ]
        if rows and len(rows[0]) == 3:
            columns = ["transaction_id", "amount_log", "is_fraud"]
        dataframe = spark.createDataFrame(
            rows,
            columns,
        )
        dataframe.write.mode("overwrite").parquet(str(root / split_name))


def _valid_rows(start_id: int) -> list[tuple]:
    return [
        (start_id, 1.0, 2.0, 0),
        (start_id + 1, 2.0, 3.0, 1),
    ]


def test_training_dataset_contract_loads_all_splits_and_builds_summary(
    spark: SparkSession, tmp_path
):
    _write_splits(
        spark,
        tmp_path,
        {
            "train": _valid_rows(1),
            "validation": _valid_rows(3),
            "test": _valid_rows(5),
        },
    )

    bundle = load_training_dataset(
        spark,
        tmp_path,
        model_feature_columns=MODEL_FEATURES,
        allow_custom_feature_columns=True,
    )

    assert bundle.summary.feature_columns == MODEL_FEATURES
    assert bundle.summary.split_summaries["train"].row_count == 2
    assert bundle.summary.split_summaries["train"].positive_target_count == 1
    assert bundle.summary.split_summaries["validation"].negative_target_count == 1
    assert bundle.summary.duplicate_transaction_id_counts == {
        "train": 0,
        "validation": 0,
        "test": 0,
    }
    assert bundle.summary.overlapping_transaction_id_counts == {
        "train__validation": 0,
        "train__test": 0,
        "validation__test": 0,
    }


def test_training_dataset_contract_rejects_missing_required_column(spark: SparkSession, tmp_path):
    _write_splits(
        spark,
        tmp_path,
        {
            "train": _valid_rows(1),
            "validation": _valid_rows(3),
            "test": [(5, 1.0, 0)],
        },
    )

    with pytest.raises(TrainingDatasetContractError, match="missing required columns"):
        load_training_dataset(
            spark,
            tmp_path,
            model_feature_columns=MODEL_FEATURES,
            allow_custom_feature_columns=True,
        )


def test_training_dataset_contract_rejects_duplicate_transaction_ids(spark: SparkSession, tmp_path):
    _write_splits(
        spark,
        tmp_path,
        {
            "train": [(1, 1.0, 2.0, 0), (1, 2.0, 3.0, 1)],
            "validation": _valid_rows(3),
            "test": _valid_rows(5),
        },
    )

    with pytest.raises(TrainingDatasetContractError, match="Duplicate transaction_id"):
        load_training_dataset(
            spark,
            tmp_path,
            model_feature_columns=MODEL_FEATURES,
            allow_custom_feature_columns=True,
        )


def test_training_dataset_contract_rejects_overlap_between_splits(spark: SparkSession, tmp_path):
    _write_splits(
        spark,
        tmp_path,
        {
            "train": _valid_rows(1),
            "validation": _valid_rows(2),
            "test": _valid_rows(5),
        },
    )

    with pytest.raises(
        TrainingDatasetContractError,
        match="overlap between splits",
    ):
        load_training_dataset(
            spark,
            tmp_path,
            model_feature_columns=MODEL_FEATURES,
            allow_custom_feature_columns=True,
        )
