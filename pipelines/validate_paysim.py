import logging

from ml.data.ingestion import read_paysim
from ml.data.spark import create_spark_session
from ml.data.validation import validate_paysim

logging.basicConfig(
    level=logging.INFO,
    format="| %(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    spark = create_spark_session()

    try:
        input_path = "data/raw/paysim.csv"

        logger.info("Reading dataset: %s", input_path)

        df = read_paysim(
            spark=spark,
            input_path=input_path,
        )

        logger.info("Schema:")
        df.printSchema()

        logger.info("Running validation")

        result = validate_paysim(df)

        logger.info("Validation result: %s", result)

        if not result.is_valid:
            raise ValueError(f"Dataset validation failed: {result}")

        logger.info("Dataset validation passed")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
