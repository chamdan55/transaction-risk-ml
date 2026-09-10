import os
import sys

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


def create_spark_session() -> SparkSession:
    python_executable = sys.executable

    os.environ["PYSPARK_PYTHON"] = python_executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_executable

    return (
        SparkSession.builder.appName("TransactionRiskML-DataPipeline")
        .master("local[4]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
