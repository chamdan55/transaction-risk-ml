from datetime import datetime

import pytest
from pyspark.sql import SparkSession

from ml.features.amount import add_amount_features


def test_amount_features_are_derived(spark: SparkSession):
    data = [
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
            0,
            0,
        ),
    ]

    columns = [
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

    df = spark.createDataFrame(data, columns)

    result = add_amount_features(df)

    row = result.select(
        "amount_log",
        "amount_to_origin_balance_ratio",
        "amount_to_destination_balance_ratio",
    ).first()

    assert row["amount_log"] == pytest.approx(4.61512051684126)
    assert row["amount_to_origin_balance_ratio"] == 0.1
    assert row["amount_to_destination_balance_ratio"] == 0.2


def test_amount_features_handle_zero_balance(spark: SparkSession):
    data = [
        (
            "tx-002",
            datetime(2026, 1, 1, 10, 0, 0),
            "CASH_OUT",
            "C001",
            "M001",
            100.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            0,
        ),
    ]

    columns = [
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

    df = spark.createDataFrame(data, columns)

    result = add_amount_features(df)

    row = result.select(
        "amount_to_origin_balance_ratio",
        "amount_to_destination_balance_ratio",
    ).first()

    assert row["amount_to_origin_balance_ratio"] == 0.0
    assert row["amount_to_destination_balance_ratio"] == 0.0
