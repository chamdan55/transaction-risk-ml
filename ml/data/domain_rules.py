from pyspark.sql import Column
from pyspark.sql import functions as F

from ml.data.constants import BALANCE_TOLERANCE

DEBIT_TRANSACTION_TYPES = (
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
)

CREDIT_TRANSACTION_TYPES = (
    "CASH_IN",
    "PAYMENT",
    "TRANSFER",
)


def build_negative_balance_condition() -> Column:
    """Return condition identifying transactions with negative balances."""
    return (
        (F.col("origin_balance_before") < 0)
        | (F.col("origin_balance_after") < 0)
        | (F.col("destination_balance_before") < 0)
        | (F.col("destination_balance_after") < 0)
    )


def build_origin_balance_inconsistency_condition() -> Column:
    """Return condition identifying inconsistent origin balances."""
    debit_expected_after = F.greatest(
        F.col("origin_balance_before") - F.col("amount"),
        F.lit(0.0),
    )

    debit_condition = F.col("transaction_type").isin(*DEBIT_TRANSACTION_TYPES) & (
        F.abs(F.col("origin_balance_after") - debit_expected_after) > BALANCE_TOLERANCE
    )

    cash_in_expected_after = F.col("origin_balance_before") + F.col("amount")

    cash_in_condition = (F.col("transaction_type") == "CASH_IN") & (
        F.abs(F.col("origin_balance_after") - cash_in_expected_after) > BALANCE_TOLERANCE
    )

    return debit_condition | cash_in_condition


def build_destination_balance_inconsistency_condition() -> Column:
    """Return condition identifying inconsistent destination balances."""
    return F.col("transaction_type").isin(*CREDIT_TRANSACTION_TYPES) & (
        F.abs(
            F.col("destination_balance_after")
            - (F.col("destination_balance_before") + F.col("amount"))
        )
        > BALANCE_TOLERANCE
    )
