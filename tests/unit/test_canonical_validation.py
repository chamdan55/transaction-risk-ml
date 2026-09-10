from datetime import datetime

from ml.data.schema import CANONICAL_TRANSACTION_SCHEMA
from ml.data.validation import validate_canonical_data


def create_valid_rows():
    return [
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
            "TRANSFER",
            "C456",
            "C789",
            250.0,
            2000.0,
            1750.0,
            1000.0,
            1250.0,
            1,
            0,
        ),
    ]


def test_valid_canonical_data(spark):
    df = spark.createDataFrame(
        create_valid_rows(),
        schema=CANONICAL_TRANSACTION_SCHEMA,
    )

    result = validate_canonical_data(df)

    assert result.is_valid is True
    assert result.row_count == 2
    assert result.null_count == 0
    assert result.invalid_transaction_id_count == 0
    assert result.duplicate_transaction_id_count == 0
    assert result.invalid_amount_count == 0
    assert result.invalid_fraud_label_count == 0
    assert result.invalid_flagged_fraud_count == 0
    assert result.invalid_transaction_type_count == 0


def test_invalid_canonical_data(spark):
    rows = [
        (
            "short-id",
            datetime(2026, 1, 1, 1, 0, 0),
            "INVALID_TYPE",
            "C123",
            "M456",
            -100.0,
            1000.0,
            1100.0,
            500.0,
            400.0,
            2,
            0,
        ),
    ]

    df = spark.createDataFrame(
        rows,
        schema=CANONICAL_TRANSACTION_SCHEMA,
    )

    result = validate_canonical_data(df)

    assert result.is_valid is False
    assert result.invalid_transaction_id_count == 1
    assert result.invalid_amount_count == 1
    assert result.invalid_fraud_label_count == 1
    assert result.invalid_transaction_type_count == 1


def test_duplicate_transaction_ids(spark):
    transaction_id = "a" * 64

    rows = [
        (
            transaction_id,
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
            transaction_id,
            datetime(2026, 1, 1, 2, 0, 0),
            "TRANSFER",
            "C456",
            "C789",
            200.0,
            2000.0,
            1800.0,
            1000.0,
            1200.0,
            0,
            0,
        ),
    ]

    df = spark.createDataFrame(
        rows,
        schema=CANONICAL_TRANSACTION_SCHEMA,
    )

    result = validate_canonical_data(df)

    assert result.is_valid is False
    assert result.duplicate_transaction_id_count == 1
