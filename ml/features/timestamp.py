from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def add_timestamp_features(df: DataFrame) -> DataFrame:
    """Add timestamp-derived features to the transaction dataset."""
    return df.withColumns(
        {
            "transaction_hour": F.hour("timestamp"),
            "transaction_day_of_week": F.dayofweek("timestamp"),
        }
    )
