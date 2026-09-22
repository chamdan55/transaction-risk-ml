import os
import sys
from collections.abc import Mapping
from typing import Any

from pyspark.sql import SparkSession

# def create_spark_session() -> SparkSession:
#     python_executable = sys.executable

#     os.environ["PYSPARK_PYTHON"] = python_executable
#     os.environ["PYSPARK_DRIVER_PYTHON"] = python_executable

#     return (
#         SparkSession.builder.appName("TransactionRiskML-DataPipeline")
#         .master("local[*]")
#         .getOrCreate()
#     )


DEFAULT_SPARK_CONFIG: dict[str, Any] = {
    "app_name": "TransactionRiskML-DataPipeline",
    "master": "local[4]",
    "driver_memory": "4g",
    "shuffle_partitions": 8,
    "default_parallelism": 4,
    "output_partitions": 4,
    "persist_intermediates": True,
}


def resolve_spark_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Merge and validate Spark/runtime settings from a pipeline configuration."""

    if config is not None and not isinstance(config, Mapping):
        raise ValueError("spark configuration must be a mapping")
    resolved = {**DEFAULT_SPARK_CONFIG, **(dict(config) if config is not None else {})}
    for key in ("app_name", "master", "driver_memory"):
        if not isinstance(resolved[key], str) or not resolved[key].strip():
            raise ValueError(f"spark.{key} must be a non-empty string")
    for key in ("shuffle_partitions", "default_parallelism", "output_partitions"):
        value = resolved[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"spark.{key} must be a positive integer")
    if not isinstance(resolved["persist_intermediates"], bool):
        raise ValueError("spark.persist_intermediates must be a boolean")
    return resolved


def create_spark_session(config: Mapping[str, Any] | None = None) -> SparkSession:
    """Create a configured Spark session with safe local defaults."""

    spark_config = resolve_spark_config(config)
    python_executable = sys.executable

    os.environ["PYSPARK_PYTHON"] = python_executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_executable

    return (
        SparkSession.builder.appName(spark_config["app_name"])
        .master(spark_config["master"])
        .config("spark.driver.memory", spark_config["driver_memory"])
        .config("spark.sql.shuffle.partitions", str(spark_config["shuffle_partitions"]))
        .config("spark.default.parallelism", str(spark_config["default_parallelism"]))
        .getOrCreate()
    )
