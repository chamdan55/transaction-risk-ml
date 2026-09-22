import json
import logging
from pathlib import Path

import yaml

from app.core.logging import setup_logging
from ml.data.spark import create_spark_session, resolve_spark_config
from ml.data.split import chronological_split, project_model_dataset, validate_split_datasets
from ml.features.schema import (
    validate_feature_schema,
    validate_feature_values,
)

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    setup_logging()
    config = load_config("configs/data.yaml")
    spark_config = resolve_spark_config(config.get("spark"))
    spark = create_spark_session(spark_config)

    try:
        processed_path = config["data"]["processed_path"]
        output_path = f"{processed_path}/features"
        input_path = f"{output_path}/audit"
        split_config = config["split"]
        output_partitions = spark_config["output_partitions"]

        logger.info("Reading feature dataset: %s", input_path)
        feature_df = spark.read.parquet(input_path)
        validate_feature_schema(feature_df)
        validate_feature_values(feature_df)

        splits = chronological_split(
            feature_df,
            train_ratio=split_config["train_ratio"],
            validation_ratio=split_config["validation_ratio"],
            test_ratio=split_config["test_ratio"],
            strategy=split_config.get("strategy", "exact"),
            quantile_relative_error=float(split_config.get("quantile_relative_error", 0.01)),
        )
        validation_result = validate_split_datasets(feature_df, splits)
        if not validation_result.is_valid:
            raise ValueError(f"Split validation failed: {validation_result}")

        for split_name, split_df in splits.items():
            split_output_path = f"{output_path}/{split_name}"
            logger.info(
                "Writing %s split: %s rows to %s",
                split_name,
                getattr(validation_result, f"{split_name}_row_count"),
                split_output_path,
            )
            project_model_dataset(split_df).coalesce(output_partitions).write.mode(
                "overwrite"
            ).parquet(split_output_path)
        Path(output_path).mkdir(parents=True, exist_ok=True)
        (Path(output_path) / "lineage.json").write_text(
            json.dumps(
                {
                    "feature_contract_version": "pre-transaction-v1",
                    "split_time_ranges": validation_result.time_ranges or {},
                    "split_strategy": split_config.get("strategy", "exact"),
                    "train_ratio": split_config["train_ratio"],
                    "validation_ratio": split_config["validation_ratio"],
                    "test_ratio": split_config["test_ratio"],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        logger.info("Feature dataset splitting completed: %s", validation_result)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
