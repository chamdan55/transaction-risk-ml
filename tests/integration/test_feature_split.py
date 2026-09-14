from datetime import datetime, timedelta

from pyspark.sql import SparkSession

from ml.data.schema import CANONICAL_TRANSACTION_SCHEMA
from ml.data.split import chronological_split, validate_split_datasets
from ml.features.pipeline import build_features
from ml.features.schema import validate_feature_schema, validate_feature_values


def test_feature_dataset_can_be_split_and_written_as_parquet(
    spark: SparkSession,
    tmp_path,
):
    start = datetime(2026, 1, 1)
    canonical_rows = [
        (
            f"tx-{index:02d}",
            start + timedelta(hours=index),
            "PAYMENT",
            "C001",
            f"M{index % 3}",
            100.0 + index,
            1000.0 + index,
            900.0,
            500.0,
            600.0 + index,
            index % 2,
            0,
        )
        for index in range(10)
    ]
    canonical_df = spark.createDataFrame(
        canonical_rows,
        schema=CANONICAL_TRANSACTION_SCHEMA,
    )
    feature_df = build_features(canonical_df)
    validate_feature_schema(feature_df)
    validate_feature_values(feature_df)

    splits = chronological_split(feature_df, 0.7, 0.15, 0.15)
    validation_result = validate_split_datasets(feature_df, splits)

    assert validation_result.is_valid
    assert validation_result.train_row_count == 7
    assert validation_result.validation_row_count == 1
    assert validation_result.test_row_count == 2

    output_root = tmp_path / "features"
    for split_name, split_df in splits.items():
        split_df.write.mode("overwrite").parquet(str(output_root / split_name))

        reloaded_df = spark.read.parquet(str(output_root / split_name))
        validate_feature_schema(reloaded_df)
        assert reloaded_df.count() == split_df.count()
