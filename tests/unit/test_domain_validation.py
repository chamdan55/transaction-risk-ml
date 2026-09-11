from datetime import datetime

from ml.data.schema import CANONICAL_TRANSACTION_SCHEMA
from ml.data.validation import validate_transaction_domain


def test_inconsistent_origin_balance(spark):
    rows = [
        (
            "a" * 64,
            datetime(2026, 1, 1, 1, 0, 0),
            "PAYMENT",
            "C123",
            "M456",
            100.0,
            1000.0,
            950.0,
            500.0,
            600.0,
            0,
            0,
        ),
    ]

    df = spark.createDataFrame(
        rows,
        schema=CANONICAL_TRANSACTION_SCHEMA,
    )

    result = validate_transaction_domain(df)

    assert result.origin_balance_mismatch_count == 1


def test_negative_balance(spark):
    rows = [
        (
            "a" * 64,
            datetime(2026, 1, 1, 1, 0, 0),
            "PAYMENT",
            "C123",
            "M456",
            100.0,
            1000.0,
            -50.0,
            500.0,
            600.0,
            0,
            0,
        ),
    ]

    df = spark.createDataFrame(
        rows,
        schema=CANONICAL_TRANSACTION_SCHEMA,
    )

    result = validate_transaction_domain(df)

    assert result.negative_balance_count == 1


def test_valid_transaction_domain(spark):
    rows = [
        (
            "a" * 64,
            datetime(2026, 1, 1, 1, 0, 0),
            "PAYMENT",
            "C123",
            "M456",
            100.0,
            1000.0,
            900.0,
            500.0,
            600.0,
            0,
            0,
        ),
        (
            "b" * 64,
            datetime(2026, 1, 1, 2, 0, 0),
            "CASH_IN",
            "C456",
            "C789",
            250.0,
            500.0,
            750.0,
            1000.0,
            1250.0,
            0,
            0,
        ),
    ]

    df = spark.createDataFrame(
        rows,
        schema=CANONICAL_TRANSACTION_SCHEMA,
    )

    result = validate_transaction_domain(df)

    assert result.negative_balance_count == 0
    assert result.origin_balance_mismatch_count == 0
    assert result.destination_balance_mismatch_count == 0
