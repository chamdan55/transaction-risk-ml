import os

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    os.environ.setdefault(
        "PYSPARK_PYTHON",
        os.environ.get("PYTHON_EXECUTABLE", "python"),
    )

    session = (
        SparkSession.builder.appName("TransactionRiskML-Test")
        .master("local[2]")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    yield session

    session.stop()
