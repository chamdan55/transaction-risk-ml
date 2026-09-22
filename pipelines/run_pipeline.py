import json
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml
from pyspark.sql import DataFrame, SparkSession
from pyspark.storagelevel import StorageLevel

from app.core.logging import setup_logging
from ml.data.ingestion import read_paysim
from ml.data.profiling import aggregate_balance_behavior, profile_balance_behavior
from ml.data.spark import create_spark_session, resolve_spark_config
from ml.data.split import chronological_split, project_model_dataset, validate_split_datasets
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


def _write_parquet(df: DataFrame, output_path: str, partitions: int) -> None:
    if partitions < 1:
        raise ValueError("output partitions must be at least 1")
    df.coalesce(partitions).write.mode("overwrite").parquet(output_path)


def run_pipeline(
    config_path: str | Path = "configs/data.yaml",
    spark: SparkSession | None = None,
) -> PipelineSummary:
    """Run the complete Sprint 1 raw-to-split pipeline."""
    config = load_config(config_path)
    owns_spark = spark is None
    spark_config = resolve_spark_config(config.get("spark"))
    spark = spark or create_spark_session(spark_config)
    output_partitions = spark_config["output_partitions"]
    persist_intermediates = spark_config["persist_intermediates"]
    split_config = config["split"]
    split_strategy = split_config.get("strategy", "exact")
    quantile_relative_error = float(split_config.get("quantile_relative_error", 0.01))
    persisted_frames: list[DataFrame] = []

    def persist_if_configured(dataframe: DataFrame) -> DataFrame:
        if persist_intermediates:
            dataframe.persist(StorageLevel.MEMORY_AND_DISK)
            persisted_frames.append(dataframe)
        return dataframe

    try:
        raw_path = config["data"]["raw_path"]
        processed_path = config["data"]["processed_path"]
        canonical_path = f"{processed_path}/canonical"
        feature_path = f"{processed_path}/features"
        audit_feature_path = f"{feature_path}/audit"
        base_timestamp = config["time"]["base_timestamp"]
        logger.info("[1/8] Reading raw dataset: %s", raw_path)
        raw_df = persist_if_configured(read_paysim(spark=spark, input_path=raw_path))
        input_row_count = raw_df.count()

        logger.info("[2/8] Validating raw dataset")
        raw_validation = validate_paysim(raw_df)
        if not raw_validation.is_valid:
            raise ValueError(f"Raw dataset validation failed: {raw_validation}")

        logger.info("[3/8] Building canonical dataset")
        canonical_df = persist_if_configured(
            transform_to_canonical(
                df=raw_df,
                base_timestamp=base_timestamp,
            )
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

        _write_parquet(canonical_df, canonical_path, output_partitions)

        logger.info("[5/8] Profiling canonical dataset")
        profile_df = aggregate_balance_behavior(profile_balance_behavior(canonical_df))
        logger.info("Balance profile: %s", profile_df.collect())

        logger.info("[6/8] Building and validating feature dataset")
        feature_df = persist_if_configured(build_features(canonical_df))
        validate_feature_schema(feature_df)
        validate_feature_values(feature_df)
        _write_parquet(feature_df, audit_feature_path, output_partitions)
        feature_row_count = feature_df.count()

        logger.info("[7/8] Splitting feature dataset chronologically")
        splits = chronological_split(
            feature_df,
            train_ratio=split_config["train_ratio"],
            validation_ratio=split_config["validation_ratio"],
            test_ratio=split_config["test_ratio"],
            strategy=split_strategy,
            quantile_relative_error=quantile_relative_error,
        )
        split_validation = validate_split_datasets(feature_df, splits)
        if not split_validation.is_valid:
            raise ValueError(f"Feature split validation failed: {split_validation}")

        for split_name, split_df in splits.items():
            model_split_df = project_model_dataset(split_df)
            _write_parquet(model_split_df, f"{feature_path}/{split_name}", output_partitions)
        Path(feature_path).mkdir(parents=True, exist_ok=True)
        (Path(feature_path) / "lineage.json").write_text(
            json.dumps(
                {
                    "feature_contract_version": "pre-transaction-v1",
                    "split_time_ranges": split_validation.time_ranges or {},
                    "split_strategy": split_strategy,
                    "train_ratio": split_config["train_ratio"],
                    "validation_ratio": split_config["validation_ratio"],
                    "test_ratio": split_config["test_ratio"],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

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
        for dataframe in reversed(persisted_frames):
            dataframe.unpersist(blocking=False)
        if owns_spark:
            spark.stop()


def main() -> None:
    setup_logging()
    run_pipeline()


if __name__ == "__main__":
    main()
