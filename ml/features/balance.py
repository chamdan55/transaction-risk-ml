from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ml.data.constants import BALANCE_TOLERANCE

DEBIT_TRANSACTION_TYPES = ("CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
CREDIT_TRANSACTION_TYPES = ("CASH_IN", "PAYMENT", "TRANSFER")


def add_balance_features(df: DataFrame) -> DataFrame:
    """Add balance-derived features."""
    origin_debit_expected_after = F.greatest(
        F.col("origin_balance_before") - F.col("amount"),
        F.lit(0.0),
    )
    origin_credit_expected_after = F.col("origin_balance_before") + F.col("amount")

    origin_mismatch = (
        F.col("transaction_type").isin(*DEBIT_TRANSACTION_TYPES)
        & (F.abs(F.col("origin_balance_after") - origin_debit_expected_after) > BALANCE_TOLERANCE)
    ) | (
        (F.col("transaction_type") == "CASH_IN")
        & (F.abs(F.col("origin_balance_after") - origin_credit_expected_after) > BALANCE_TOLERANCE)
    )

    destination_mismatch = F.col("transaction_type").isin(*CREDIT_TRANSACTION_TYPES) & (
        F.abs(
            F.col("destination_balance_after")
            - (F.col("destination_balance_before") + F.col("amount"))
        )
        > BALANCE_TOLERANCE
    )

    return df.withColumns(
        {
            "origin_balance_delta": (
                F.col("origin_balance_after") - F.col("origin_balance_before")
            ),
            "origin_balance_change_ratio": F.when(
                F.col("origin_balance_before") > 0,
                (F.col("origin_balance_after") - F.col("origin_balance_before"))
                / F.col("origin_balance_before"),
            ).otherwise(0.0),
            "destination_balance_delta": (
                F.col("destination_balance_after") - F.col("destination_balance_before")
            ),
            "destination_balance_change_ratio": F.when(
                F.col("destination_balance_before") > 0,
                (F.col("destination_balance_after") - F.col("destination_balance_before"))
                / F.col("destination_balance_before"),
            ).otherwise(0.0),
            "origin_balance_mismatch": origin_mismatch.cast("int"),
            "destination_balance_mismatch": destination_mismatch.cast("int"),
            "is_zero_origin_balance_before": ((F.col("origin_balance_before") == 0).cast("int")),
            "is_zero_destination_balance_before": (
                (F.col("destination_balance_before") == 0).cast("int")
            ),
            "origin_balance_depleted": ((F.col("origin_balance_after") == 0).cast("int")),
            "destination_balance_increased": (
                (F.col("destination_balance_after") > F.col("destination_balance_before")).cast(
                    "int"
                )
            ),
        }
    )
