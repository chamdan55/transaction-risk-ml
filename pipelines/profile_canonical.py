import logging
from pathlib import Path

from ml.data.profiling import (
    aggregate_balance_behavior,
    profile_balance_behavior,
)
from ml.data.spark import create_spark_session

logging.basicConfig(
    level=logging.INFO,
    format="| %(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

CANONICAL_DATASET_PATH = Path("data/processed/canonical")


def main() -> None:
    spark = create_spark_session()

    logger.info("Reading canonical dataset: %s", CANONICAL_DATASET_PATH)

    df = spark.read.parquet(str(CANONICAL_DATASET_PATH))

    logger.info("Profiling balance behavior by transaction type")

    profiled_df = profile_balance_behavior(df)

    profile_df = aggregate_balance_behavior(profiled_df)

    logger.info("Balance behavior profile:")
    profile_df.show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
