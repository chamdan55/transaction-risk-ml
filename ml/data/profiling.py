from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ml.data.domain_rules import (
    build_destination_balance_inconsistency_condition,
    build_origin_balance_inconsistency_condition,
)

BALANCE_TOLERANCE = 0.01


def profile_balance_consistency_by_type(
    df: DataFrame,
) -> DataFrame:
    """Profile balance consistency metrics by transaction type."""
    origin_inconsistent = build_origin_balance_inconsistency_condition()
    destination_inconsistent = build_destination_balance_inconsistency_condition()

    return (
        df.groupBy("transaction_type")
        .agg(
            F.count("*").alias("transaction_count"),
            F.sum(F.when(origin_inconsistent, 1).otherwise(0)).alias("inconsistent_origin_count"),
            F.sum(F.when(destination_inconsistent, 1).otherwise(0)).alias(
                "inconsistent_destination_count"
            ),
        )
        .withColumn(
            "origin_inconsistency_rate",
            F.col("inconsistent_origin_count") / F.col("transaction_count"),
        )
        .withColumn(
            "destination_inconsistency_rate",
            F.col("inconsistent_destination_count") / F.col("transaction_count"),
        )
        .orderBy("transaction_type")
    )


def profile_balance_behavior(df: DataFrame) -> DataFrame:
    origin_delta = F.col("origin_balance_after") - F.col("origin_balance_before")

    destination_delta = F.col("destination_balance_after") - F.col("destination_balance_before")

    amount = F.col("amount")

    origin_expected_delta = (
        F.when(
            F.col("transaction_type") == "CASH_IN",
            amount,
        )
        .when(
            F.col("transaction_type").isin(
                "CASH_OUT",
                "DEBIT",
                "PAYMENT",
                "TRANSFER",
            ),
            -amount,
        )
        .otherwise(F.lit(None))
    )

    origin_exact_match = F.abs(origin_delta - origin_expected_delta) <= BALANCE_TOLERANCE

    origin_floor_match = (
        F.col("transaction_type").isin(
            "CASH_OUT",
            "DEBIT",
            "PAYMENT",
            "TRANSFER",
        )
        & (F.col("origin_balance_before") < amount)
        & (F.abs(F.col("origin_balance_after")) <= BALANCE_TOLERANCE)
    )

    destination_plus_amount_match = F.abs(destination_delta - amount) <= BALANCE_TOLERANCE

    return (
        df.withColumn("origin_delta", origin_delta)
        .withColumn("destination_delta", destination_delta)
        .withColumn("origin_expected_delta", origin_expected_delta)
        .withColumn("origin_exact_match", origin_exact_match)
        .withColumn("origin_floor_match", origin_floor_match)
        .withColumn(
            "destination_plus_amount_match",
            destination_plus_amount_match,
        )
    )


def aggregate_balance_behavior(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("transaction_type")
        .agg(
            F.count("*").alias("transaction_count"),
            F.sum(F.col("origin_exact_match").cast("int")).alias("origin_exact_match_count"),
            F.sum(F.col("origin_floor_match").cast("int")).alias("origin_floor_match_count"),
            F.sum(F.col("destination_plus_amount_match").cast("int")).alias(
                "destination_plus_amount_match_count"
            ),
        )
        .withColumn(
            "origin_mismatch_count",
            F.col("transaction_count")
            - F.col("origin_exact_match_count")
            - F.col("origin_floor_match_count"),
        )
        .withColumn(
            "origin_mismatch_rate",
            F.col("origin_mismatch_count") / F.col("transaction_count"),
        )
        .withColumn(
            "destination_mismatch_count",
            F.col("transaction_count") - F.col("destination_plus_amount_match_count"),
        )
        .withColumn(
            "destination_mismatch_rate",
            F.col("destination_mismatch_count") / F.col("transaction_count"),
        )
        .orderBy("transaction_type")
    )
