from datetime import datetime

from pyspark.sql import SparkSession

from ml.features.pipeline import build_features
from ml.features.schema import FEATURE_SCHEMA

CANONICAL_COLUMNS = [
    "transaction_id",
    "timestamp",
    "transaction_type",
    "origin_account_id",
    "destination_account_id",
    "amount",
    "origin_balance_before",
    "origin_balance_after",
    "destination_balance_before",
    "destination_balance_after",
    "is_fraud",
    "is_flagged_fraud",
]


def _canonical_rows(future_amount: float, first_is_fraud: int) -> list[tuple]:
    return [
        (
            "tx-001",
            datetime(2026, 1, 1, 10, 0, 0),
            "PAYMENT",
            "C001",
            "M001",
            100.0,
            1000.0,
            900.0,
            500.0,
            600.0,
            first_is_fraud,
            0,
        ),
        (
            "tx-002",
            datetime(2026, 1, 1, 11, 0, 0),
            "TRANSFER",
            "C001",
            "C002",
            future_amount,
            900.0,
            0.0,
            0.0,
            future_amount,
            1,
            0,
        ),
    ]


def _build_feature_df(spark: SparkSession, rows: list[tuple]):
    return build_features(spark.createDataFrame(rows, CANONICAL_COLUMNS))


def test_features_do_not_use_future_transaction_or_target_label(
    spark: SparkSession,
):
    baseline = _build_feature_df(spark, _canonical_rows(200.0, 0))
    changed_future = _build_feature_df(spark, _canonical_rows(999999.0, 0))
    changed_target = _build_feature_df(spark, _canonical_rows(200.0, 1))

    feature_columns = [
        column
        for column in FEATURE_SCHEMA.names
        if column not in {"transaction_id", "is_fraud", "is_flagged_fraud"}
    ]

    baseline_first = baseline.orderBy("timestamp").select(feature_columns).first()
    future_first = changed_future.orderBy("timestamp").select(feature_columns).first()
    target_first = changed_target.orderBy("timestamp").select(feature_columns).first()

    assert baseline_first == future_first
    assert baseline_first == target_first
