import logging
from dataclasses import dataclass
from pathlib import Path

import yaml
from pyspark.sql import DataFrame, SparkSession

from app.core.logging import setup_logging
from ml.data.ingestion import read_paysim
from ml.data.profiling import aggregate_balance_behavior, profile_balance_behavior
from ml.data.spark import create_spark_session
from ml.data.split import chronological_split, validate_split_datasets
from ml.data.transformation import transform_to_canonical
from ml.data.validation import (
    validate_canonical_data,
    validate_paysim,
    validate_transaction_domain,
)
from ml.features.pipeline import build_features
from ml.features.schema import validate_feature_schema, validate_feature_values

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineSummary:
    input_row_count: int
    canonical_row_count: int
    feature_row_count: int
    train_row_count: int
    validation_row_count: int
    test_row_count: int
    feature_column_count: int
    target_column: str
    feature_output_path: str


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def _write_parquet(df: DataFrame, output_path: str, partitions: int = 4) -> None:
    df.coalesce(partitions).write.mode("overwrite").parquet(output_path)


def run_pipeline(
    config_path: str | Path = "configs/data.yaml",
    spark: SparkSession | None = None,
) -> PipelineSummary:
    """Run the complete Sprint 1 raw-to-split pipeline."""
    config = load_config(config_path)
    owns_spark = spark is None
    spark = spark or create_spark_session()

    try:
        raw_path = config["data"]["raw_path"]
        processed_path = config["data"]["processed_path"]
        canonical_path = f"{processed_path}/canonical"
        feature_path = f"{processed_path}/features"
        base_timestamp = config["time"]["base_timestamp"]
        split_config = config["split"]

        logger.info("[1/8] Reading raw dataset: %s", raw_path)
        raw_df = read_paysim(spark=spark, input_path=raw_path)
        input_row_count = raw_df.count()

        logger.info("[2/8] Validating raw dataset")
        raw_validation = validate_paysim(raw_df)
        if not raw_validation.is_valid:
            raise ValueError(f"Raw dataset validation failed: {raw_validation}")

        logger.info("[3/8] Building canonical dataset")
        canonical_df = transform_to_canonical(
            df=raw_df,
            base_timestamp=base_timestamp,
        )
        canonical_row_count = canonical_df.count()

        logger.info("[4/8] Validating canonical dataset")
        canonical_validation = validate_canonical_data(canonical_df)
        if not canonical_validation.is_valid:
            raise ValueError(f"Canonical dataset validation failed: {canonical_validation}")

        domain_validation = validate_transaction_domain(canonical_df)
        logger.info("Domain validation result: %s", domain_validation)
        if domain_validation.negative_balance_count > 0:
            raise ValueError(
                "Canonical dataset contains negative balances: "
                f"{domain_validation.negative_balance_count} rows"
            )

        _write_parquet(canonical_df, canonical_path)

        logger.info("[5/8] Profiling canonical dataset")
        profile_df = aggregate_balance_behavior(profile_balance_behavior(canonical_df))
        logger.info("Balance profile: %s", profile_df.collect())

        logger.info("[6/8] Building and validating feature dataset")
        feature_df = build_features(canonical_df)
        validate_feature_schema(feature_df)
        validate_feature_values(feature_df)
        _write_parquet(feature_df, feature_path)
        feature_row_count = feature_df.count()

        logger.info("[7/8] Splitting feature dataset chronologically")
        splits = chronological_split(
            feature_df,
            train_ratio=split_config["train_ratio"],
            validation_ratio=split_config["validation_ratio"],
            test_ratio=split_config["test_ratio"],
        )
        split_validation = validate_split_datasets(feature_df, splits)
        if not split_validation.is_valid:
            raise ValueError(f"Feature split validation failed: {split_validation}")

        for split_name, split_df in splits.items():
            _write_parquet(split_df, f"{feature_path}/{split_name}")

        logger.info("[8/8] Pipeline summary")
        summary = PipelineSummary(
            input_row_count=input_row_count,
            canonical_row_count=canonical_row_count,
            feature_row_count=feature_row_count,
            train_row_count=split_validation.train_row_count,
            validation_row_count=split_validation.validation_row_count,
            test_row_count=split_validation.test_row_count,
            feature_column_count=len(feature_df.columns),
            target_column="is_fraud",
            feature_output_path=feature_path,
        )
        logger.info("Pipeline completed successfully: %s", summary)
        return summary
    finally:
        if owns_spark:
            spark.stop()


def main() -> None:
    setup_logging()
    run_pipeline()


if __name__ == "__main__":
    main()
