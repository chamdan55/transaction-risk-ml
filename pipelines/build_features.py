import logging

import yaml

from app.core.logging import setup_logging
from ml.data.spark import create_spark_session, resolve_spark_config
from ml.data.validation import validate_canonical_data
from ml.features.pipeline import build_features
from ml.features.schema import validate_feature_schema, validate_feature_values

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
        input_path = f"{processed_path}/canonical"
        output_path = f"{processed_path}/features/audit"

        logger.info("Reading canonical dataset: %s", input_path)
        canonical_df = spark.read.parquet(input_path)
        validation_result = validate_canonical_data(canonical_df)

        if not validation_result.is_valid:
            raise ValueError(f"Canonical validation failed: {validation_result}")

        logger.info("Building Step 5 features")
        feature_df = build_features(canonical_df)
        validate_feature_schema(feature_df)
        validate_feature_values(feature_df)

        logger.info("Writing feature dataset: %s", output_path)
        output_partitions = spark_config["output_partitions"]
        feature_df.coalesce(output_partitions).write.mode("overwrite").parquet(output_path)
        logger.info("Feature dataset written successfully")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
