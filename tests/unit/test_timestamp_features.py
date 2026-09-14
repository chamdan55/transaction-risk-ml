from datetime import datetime

from pyspark.sql import SparkSession

from ml.features.timestamp import add_timestamp_features


def test_timestamp_features_are_derived(spark: SparkSession):
    data = [
        (datetime(2026, 1, 1, 10, 30, 0),),
        (datetime(2026, 1, 4, 23, 45, 0),),
    ]

    df = spark.createDataFrame(
        data,
        ["timestamp"],
    )

    result = add_timestamp_features(df)

    rows = result.select(
        "transaction_hour",
        "transaction_day_of_week",
    ).collect()

    assert rows[0]["transaction_hour"] == 10
    assert rows[0]["transaction_day_of_week"] == 5

    assert rows[1]["transaction_hour"] == 23
    assert rows[1]["transaction_day_of_week"] == 1
