from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

SECONDS_PER_HOUR = 60 * 60
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR
SECONDS_PER_30_DAYS = 30 * SECONDS_PER_DAY


def add_behavior_features(df: DataFrame) -> DataFrame:
    """Add historical account behavior features without current-row leakage."""
    event_time = F.unix_timestamp("timestamp")

    origin_window = Window.partitionBy("origin_account_id").orderBy(event_time)
    last_hour_window = origin_window.rangeBetween(-SECONDS_PER_HOUR, -1)
    last_day_window = origin_window.rangeBetween(-SECONDS_PER_DAY, -1)
    last_30_days_window = origin_window.rangeBetween(-SECONDS_PER_30_DAYS, -1)

    return df.withColumns(
        {
            "transactions_last_1h": F.count("transaction_id").over(last_hour_window).cast("int"),
            "transactions_last_24h": F.count("transaction_id").over(last_day_window).cast("int"),
            "amount_sum_last_24h": F.coalesce(
                F.sum("amount").over(last_day_window),
                F.lit(0.0),
            ),
            "unique_destinations_last_30d": F.coalesce(
                F.size(F.collect_set("destination_account_id").over(last_30_days_window)),
                F.lit(0),
            ).cast("int"),
        }
    )
