import os
import sys

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    os.environ["PYSPARK_PYTHON"] = sys.executable

    spark = (
        SparkSession.builder.appName("transaction-risk-ml-tests")
        .master("local[2]")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )

    yield spark

    spark.stop()
