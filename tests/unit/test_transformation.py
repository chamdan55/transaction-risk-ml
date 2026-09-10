from pyspark.sql import SparkSession

from ml.data.schema import PAYSIM_SCHEMA
from ml.data.transformation import transform_to_canonical


def test_transform_to_canonical(spark: SparkSession):
    data = [
        (
            1,
            "PAYMENT",
            100.0,
            "C123",
            1000.0,
            900.0,
            "M123",
            500.0,
            600.0,
            0,
            0,
        ),
    ]

    df = spark.createDataFrame(
        data,
        schema=PAYSIM_SCHEMA,
    )

    result = transform_to_canonical(
        df=df,
        base_timestamp="2026-01-01T00:00:00",
    )

    row = result.first()

    assert row.transaction_type == "PAYMENT"
    assert row.origin_account_id == "C123"
    assert row.destination_account_id == "M123"
    assert row.amount == 100.0
    assert row.is_fraud == 0
    assert row.is_flagged_fraud == 0

    assert str(row.timestamp) == "2026-01-01 01:00:00"
    assert len(row.transaction_id) == 64


def test_canonical_columns(spark: SparkSession):
    data = [
        (
            1,
            "PAYMENT",
            100.0,
            "C123",
            1000.0,
            900.0,
            "M123",
            500.0,
            600.0,
            0,
            0,
        ),
    ]

    df = spark.createDataFrame(
        data,
        schema=PAYSIM_SCHEMA,
    )

    result = transform_to_canonical(
        df=df,
        base_timestamp="2026-01-01T00:00:00",
    )

    expected_columns = [
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

    assert result.columns == expected_columns
