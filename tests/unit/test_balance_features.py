from datetime import datetime

import pytest
from pyspark.sql import SparkSession

from ml.features.balance import add_balance_features


def test_balance_features_are_derived(spark: SparkSession):
    data = [
        (
            "tx-001",
            datetime(2026, 1, 1, 10, 0, 0),
            "PAYMENT",
            "C001",
            "M001",
            300.0,
            1000.0,
            700.0,
            500.0,
            800.0,
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

    result = add_balance_features(df)

    row = result.select(
        "origin_balance_delta",
        "destination_balance_delta",
        "origin_balance_change_ratio",
        "destination_balance_change_ratio",
        "origin_balance_depleted",
        "destination_balance_increased",
    ).first()

    assert row["origin_balance_delta"] == -300.0
    assert row["destination_balance_delta"] == 300.0
    assert row["origin_balance_change_ratio"] == pytest.approx(-0.3)
    assert row["destination_balance_change_ratio"] == pytest.approx(0.6)
    assert row["origin_balance_depleted"] == 0
    assert row["destination_balance_increased"] == 1


def test_balance_features_handle_zero_origin_balance(
    spark: SparkSession,
):
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
            500.0,
            400.0,
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

    result = add_balance_features(df)

    row = result.select(
        "origin_balance_delta",
        "destination_balance_delta",
        "origin_balance_change_ratio",
    ).first()

    assert row["origin_balance_delta"] == 0.0
    assert row["destination_balance_delta"] == -100.0
    assert row["origin_balance_change_ratio"] == 0.0
