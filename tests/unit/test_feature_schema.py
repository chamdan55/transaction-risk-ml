from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from ml.features.schema import FEATURE_SCHEMA, validate_feature_schema


def test_feature_schema_is_defined():
    expected_columns = [
        "transaction_id",
        "timestamp",
        "transaction_type",
        "origin_account_id",
        "destination_account_id",
        "amount",
        "amount_log",
        "origin_balance_before",
        "origin_balance_after",
        "origin_balance_delta",
        "origin_balance_change_ratio",
        "amount_to_origin_balance_ratio",
        "destination_balance_before",
        "destination_balance_after",
        "destination_balance_delta",
        "destination_balance_change_ratio",
        "amount_to_destination_balance_ratio",
        "origin_balance_mismatch",
        "destination_balance_mismatch",
        "is_zero_origin_balance_before",
        "is_zero_destination_balance_before",
        "origin_balance_depleted",
        "destination_balance_increased",
        "transaction_hour",
        "transaction_day_of_week",
        "is_cash_in",
        "is_cash_out",
        "is_debit",
        "is_payment",
        "is_transfer",
        "transactions_last_1h",
        "transactions_last_24h",
        "amount_sum_last_24h",
        "unique_destinations_last_30d",
        "is_fraud",
        "is_flagged_fraud",
    ]

    assert FEATURE_SCHEMA.names == expected_columns


def test_valid_feature_schema(spark: SparkSession):
    df = spark.createDataFrame(
        [
            (
                "tx-1",
                datetime(2026, 1, 1, 10, 0, 0),  # "2026-01-01 10:00:00"
                "PAYMENT",
                "C1",
                "M1",
                100.0,
                4.61512051684126,
                1000.0,
                900.0,
                -100.0,
                -0.1,
                0.1,
                500.0,
                600.0,
                100.0,
                0.2,
                0.2,
                0,
                0,
                0,
                0,
                0,
                1,
                10,
                3,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0.0,
                0,
                0,
                0,
            )
        ],
        schema=StructType(
            [
                StructField("transaction_id", StringType(), False),
                StructField("timestamp", TimestampType(), False),
                StructField("transaction_type", StringType(), False),
                StructField("origin_account_id", StringType(), False),
                StructField("destination_account_id", StringType(), False),
                StructField("amount", DoubleType(), False),
                StructField("amount_log", DoubleType(), False),
                StructField("origin_balance_before", DoubleType(), False),
                StructField("origin_balance_after", DoubleType(), False),
                StructField("origin_balance_delta", DoubleType(), False),
                StructField("origin_balance_change_ratio", DoubleType(), False),
                StructField("amount_to_origin_balance_ratio", DoubleType(), True),
                StructField("destination_balance_before", DoubleType(), False),
                StructField("destination_balance_after", DoubleType(), False),
                StructField("destination_balance_delta", DoubleType(), False),
                StructField("destination_balance_change_ratio", DoubleType(), False),
                StructField("amount_to_destination_balance_ratio", DoubleType(), True),
                StructField("origin_balance_mismatch", IntegerType(), False),
                StructField("destination_balance_mismatch", IntegerType(), False),
                StructField("is_zero_origin_balance_before", IntegerType(), False),
                StructField("is_zero_destination_balance_before", IntegerType(), False),
                StructField("origin_balance_depleted", IntegerType(), False),
                StructField("destination_balance_increased", IntegerType(), False),
                StructField("transaction_hour", IntegerType(), False),
                StructField("transaction_day_of_week", IntegerType(), False),
                StructField("is_cash_in", IntegerType(), False),
                StructField("is_cash_out", IntegerType(), False),
                StructField("is_debit", IntegerType(), False),
                StructField("is_payment", IntegerType(), False),
                StructField("is_transfer", IntegerType(), False),
                StructField("transactions_last_1h", IntegerType(), False),
                StructField("transactions_last_24h", IntegerType(), False),
                StructField("amount_sum_last_24h", DoubleType(), False),
                StructField("unique_destinations_last_30d", IntegerType(), False),
                StructField("is_fraud", IntegerType(), False),
                StructField("is_flagged_fraud", IntegerType(), False),
            ]
        ),
    )

    validate_feature_schema(df)
