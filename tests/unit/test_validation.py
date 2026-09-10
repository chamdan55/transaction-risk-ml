from pyspark.sql import SparkSession

from ml.data.schema import PAYSIM_SCHEMA
from ml.data.validation import validate_paysim


def test_valid_paysim_data(spark: SparkSession):
    data = [
        (
            1,
            "PAYMENT",
            100.0,
            "C123",
            1000.0,
            900.0,
            "M123",
            500.0,
            600.0,
            0,
            0,
        ),
        (
            2,
            "TRANSFER",
            200.0,
            "C456",
            1000.0,
            800.0,
            "M456",
            300.0,
            500.0,
            1,
            0,
        ),
    ]

    df = spark.createDataFrame(
        data,
        schema=PAYSIM_SCHEMA,
    )

    result = validate_paysim(df)

    assert result.is_valid is True
    assert result.row_count == 2
    assert result.null_count == 0
    assert result.invalid_amount_count == 0
    assert result.invalid_fraud_label_count == 0
    assert result.invalid_flagged_fraud_count == 0
    assert result.invalid_step_count == 0


def test_invalid_amount_is_detected(spark: SparkSession):
    data = [
        (
            1,
            "PAYMENT",
            -100.0,
            "C123",
            1000.0,
            900.0,
            "M123",
            500.0,
            600.0,
            0,
            0,
        ),
    ]

    df = spark.createDataFrame(
        data,
        schema=PAYSIM_SCHEMA,
    )

    result = validate_paysim(df)

    assert result.is_valid is False
    assert result.invalid_amount_count == 1
