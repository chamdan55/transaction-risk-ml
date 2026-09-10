from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from ml.data.schema import PAYSIM_SCHEMA


def read_paysim(
    spark: SparkSession,
    input_path: str | Path,
) -> DataFrame:
    path = str(input_path)

    return spark.read.option("header", True).schema(PAYSIM_SCHEMA).csv(path)
