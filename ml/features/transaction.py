from pyspark.sql import DataFrame
from pyspark.sql import functions as F

TRANSACTION_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")


def add_transaction_features(df: DataFrame) -> DataFrame:
    """Add transaction-type indicator features."""
    return df.withColumns(
        {
            f"is_{transaction_type.lower()}": (
                (F.col("transaction_type") == transaction_type).cast("int")
            )
            for transaction_type in TRANSACTION_TYPES
        }
    )
