from datetime import datetime

import pytest
from pyspark.sql import Row
from pyspark.sql.types import StringType, StructField, StructType

from ml.data.schema import CANONICAL_TRANSACTION_SCHEMA
from ml.data.validation import validate_canonical_schema


def test_valid_canonical_schema(spark):
    data = [
        Row(
            transaction_id="a" * 64,
            timestamp=datetime(2026, 1, 1, 1, 0, 0),
            transaction_type="PAYMENT",
            origin_account_id="C123",
            destination_account_id="M456",
            amount=100.0,
            origin_balance_before=1000.0,
            origin_balance_after=900.0,
            destination_balance_before=500.0,
            destination_balance_after=600.0,
            is_fraud=0,
            is_flagged_fraud=0,
        )
    ]

    df = spark.createDataFrame(
        data,
        schema=CANONICAL_TRANSACTION_SCHEMA,
    )

    validate_canonical_schema(df)


def test_invalid_canonical_schema(spark):
    invalid_schema = StructType(
        [
            StructField("transaction_id", StringType(), nullable=False),
        ]
    )

    df = spark.createDataFrame(
        [("abc",)],
        schema=invalid_schema,
    )

    with pytest.raises(ValueError, match="Canonical schema mismatch"):
        validate_canonical_schema(df)


def test_nullable_metadata_does_not_fail_schema_validation(spark):
    df = spark.createDataFrame(
        [
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
            )
        ],
        schema=CANONICAL_TRANSACTION_SCHEMA,
    )

    nullable_schema = StructType(
        [
            StructField(
                field.name,
                field.dataType,
                nullable=True,
            )
            for field in CANONICAL_TRANSACTION_SCHEMA.fields
        ]
    )

    nullable_df = spark.createDataFrame(
        df.rdd,
        schema=nullable_schema,
    )

    validate_canonical_schema(nullable_df)
