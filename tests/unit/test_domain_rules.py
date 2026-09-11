from pyspark.sql import SparkSession

from ml.data.domain_rules import (
    build_origin_balance_inconsistency_condition,
)


def test_origin_balance_exact_debit_is_valid(spark: SparkSession):
    data = [
        (
            "CASH_OUT",
            1000.0,
            800.0,
            200.0,
        ),
    ]

    df = spark.createDataFrame(
        data,
        [
            "transaction_type",
            "origin_balance_before",
            "origin_balance_after",
            "amount",
        ],
    )

    invalid_count = df.filter(build_origin_balance_inconsistency_condition()).count()

    assert invalid_count == 0


def test_origin_balance_floor_to_zero_is_valid(spark: SparkSession):
    data = [
        (
            "CASH_OUT",
            100.0,
            0.0,
            200.0,
        ),
    ]

    df = spark.createDataFrame(
        data,
        [
            "transaction_type",
            "origin_balance_before",
            "origin_balance_after",
            "amount",
        ],
    )

    invalid_count = df.filter(build_origin_balance_inconsistency_condition()).count()

    assert invalid_count == 0


def test_origin_balance_inconsistency_is_detected(spark: SparkSession):
    data = [
        (
            "CASH_OUT",
            1000.0,
            900.0,
            200.0,
        ),
    ]

    df = spark.createDataFrame(
        data,
        [
            "transaction_type",
            "origin_balance_before",
            "origin_balance_after",
            "amount",
        ],
    )

    invalid_count = df.filter(build_origin_balance_inconsistency_condition()).count()

    assert invalid_count == 1


def test_cash_in_origin_balance_is_valid(spark: SparkSession):
    data = [
        (
            "CASH_IN",
            1000.0,
            1200.0,
            200.0,
        ),
    ]

    df = spark.createDataFrame(
        data,
        [
            "transaction_type",
            "origin_balance_before",
            "origin_balance_after",
            "amount",
        ],
    )

    invalid_count = df.filter(build_origin_balance_inconsistency_condition()).count()

    assert invalid_count == 0
