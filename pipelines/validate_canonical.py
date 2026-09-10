import logging

from ml.data.spark import create_spark_session
from ml.data.validation import validate_canonical_data

logging.basicConfig(
    level=logging.INFO,
    format="| %(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    input_path = "data/processed/canonical"

    spark = create_spark_session()

    try:
        logger.info("Reading canonical dataset: %s", input_path)

        df = spark.read.parquet(input_path)

        logger.info("Validating canonical dataset")

        result = validate_canonical_data(df)

        logger.info("Canonical validation result: %s", result)

        if not result.is_valid:
            raise ValueError(f"Canonical dataset validation failed: {result}")

        logger.info("Canonical dataset validation passed")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
