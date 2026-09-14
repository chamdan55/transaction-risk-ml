from datetime import datetime

from pyspark.sql import SparkSession

from ml.data.schema import CANONICAL_TRANSACTION_SCHEMA
from ml.features.pipeline import build_features
from ml.features.schema import validate_feature_schema, validate_feature_values


def test_feature_output_can_be_read_back_by_spark(
    spark: SparkSession,
    tmp_path,
):
    canonical_rows = [
        (
            "tx-001",
            datetime(2026, 1, 1, 10, 0, 0),
            "PAYMENT",
            "C001",
            "M001",
            100.0,
            1000.0,
            900.0,
            500.0,
            600.0,
            0,
            0,
        ),
        (
            "tx-002",
            datetime(2026, 1, 1, 11, 0, 0),
            "TRANSFER",
            "C001",
            "C002",
            200.0,
            900.0,
            700.0,
            0.0,
            200.0,
            1,
            0,
        ),
    ]
    canonical_df = spark.createDataFrame(
        canonical_rows,
        schema=CANONICAL_TRANSACTION_SCHEMA,
    )
    feature_df = build_features(canonical_df)
    validate_feature_schema(feature_df)
    validate_feature_values(feature_df)

    output_path = str(tmp_path / "features")
    feature_df.write.mode("overwrite").parquet(output_path)

    reloaded_df = spark.read.parquet(output_path)

    validate_feature_schema(reloaded_df)
    validate_feature_values(reloaded_df)
    assert reloaded_df.count() == len(canonical_rows)
