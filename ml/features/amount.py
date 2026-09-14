from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def add_amount_features(df: DataFrame) -> DataFrame:
    """Add transaction amount-derived features."""
    return df.withColumns(
        {
            "amount_log": F.log1p("amount"),
            "amount_to_origin_balance_ratio": F.when(
                F.col("origin_balance_before") > 0,
                F.col("amount") / F.col("origin_balance_before"),
            ).otherwise(0.0),
            "amount_to_destination_balance_ratio": F.when(
                F.col("destination_balance_before") > 0,
                F.col("amount") / F.col("destination_balance_before"),
            ).otherwise(0.0),
        }
    )
