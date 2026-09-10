import logging

import yaml

from ml.data.ingestion import read_paysim
from ml.data.spark import create_spark_session
from ml.data.transformation import transform_to_canonical
from ml.data.validation import validate_paysim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    config = load_config("configs/data.yaml")

    spark = create_spark_session()

    try:
        input_path = config["data"]["raw_path"]
        output_path = config["data"]["processed_path"]
        base_timestamp = config["time"]["base_timestamp"]

        logger.info("Reading dataset: %s", input_path)

        df = read_paysim(
            spark=spark,
            input_path=input_path,
        )

        logger.info("Validating input dataset")

        validation_result = validate_paysim(df)

        if not validation_result.is_valid:
            raise ValueError(f"Dataset validation failed: {validation_result}")

        logger.info(
            "Input validation passed: %s",
            validation_result,
        )

        logger.info("Transforming dataset to canonical schema")

        canonical_df = transform_to_canonical(
            df=df,
            base_timestamp=base_timestamp,
        )

        logger.info("Canonical schema:")
        canonical_df.printSchema()

        logger.info("Writing canonical dataset to: %s", output_path)

        canonical_df = canonical_df.coalesce(4)
        canonical_df.write.mode("overwrite").parquet(f"{output_path}/canonical")

        logger.info("Canonical dataset written successfully")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
