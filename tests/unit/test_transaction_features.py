from pyspark.sql import SparkSession

from ml.features.transaction import add_transaction_features


def test_transaction_type_indicators(spark: SparkSession):
    df = spark.createDataFrame(
        [("PAYMENT",), ("TRANSFER",)],
        ["transaction_type"],
    )

    rows = add_transaction_features(df).collect()

    assert rows[0]["is_payment"] == 1
    assert rows[0]["is_transfer"] == 0
    assert rows[1]["is_payment"] == 0
    assert rows[1]["is_transfer"] == 1
