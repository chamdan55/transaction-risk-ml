from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

FEATURE_SCHEMA = StructType(
    [
        StructField("transaction_id", StringType(), nullable=False),
        StructField("timestamp", TimestampType(), nullable=False),
        StructField("transaction_type", StringType(), nullable=False),
        StructField("origin_account_id", StringType(), nullable=False),
        StructField("destination_account_id", StringType(), nullable=False),
        StructField("amount", DoubleType(), nullable=False),
        StructField("amount_log", DoubleType(), nullable=False),
        StructField("origin_balance_before", DoubleType(), nullable=False),
        StructField("origin_balance_after", DoubleType(), nullable=False),
        StructField("origin_balance_delta", DoubleType(), nullable=False),
        StructField("origin_balance_change_ratio", DoubleType(), nullable=False),
        StructField("amount_to_origin_balance_ratio", DoubleType(), nullable=True),
        StructField("destination_balance_before", DoubleType(), nullable=False),
        StructField("destination_balance_after", DoubleType(), nullable=False),
        StructField("destination_balance_delta", DoubleType(), nullable=False),
        StructField("destination_balance_change_ratio", DoubleType(), nullable=False),
        StructField("amount_to_destination_balance_ratio", DoubleType(), nullable=True),
        StructField("origin_balance_mismatch", IntegerType(), nullable=False),
        StructField("destination_balance_mismatch", IntegerType(), nullable=False),
        StructField("is_zero_origin_balance_before", IntegerType(), nullable=False),
        StructField("is_zero_destination_balance_before", IntegerType(), nullable=False),
        StructField("origin_balance_depleted", IntegerType(), nullable=False),
        StructField("destination_balance_increased", IntegerType(), nullable=False),
        StructField("transaction_hour", IntegerType(), nullable=False),
        StructField("transaction_day_of_week", IntegerType(), nullable=False),
        StructField("is_cash_in", IntegerType(), nullable=False),
        StructField("is_cash_out", IntegerType(), nullable=False),
        StructField("is_debit", IntegerType(), nullable=False),
        StructField("is_payment", IntegerType(), nullable=False),
        StructField("is_transfer", IntegerType(), nullable=False),
        StructField("transactions_last_1h", IntegerType(), nullable=False),
        StructField("transactions_last_24h", IntegerType(), nullable=False),
        StructField("amount_sum_last_24h", DoubleType(), nullable=False),
        StructField("unique_destinations_last_30d", IntegerType(), nullable=False),
        StructField("is_fraud", IntegerType(), nullable=False),
        StructField("is_flagged_fraud", IntegerType(), nullable=False),
    ]
)


def validate_feature_schema(df: DataFrame) -> None:
    expected_fields = FEATURE_SCHEMA.fields
    actual_fields = df.schema.fields

    if len(actual_fields) != len(expected_fields):
        raise ValueError(
            "Feature schema mismatch: "
            f"expected {len(expected_fields)} fields, "
            f"got {len(actual_fields)}."
        )

    for expected, actual in zip(
        expected_fields,
        actual_fields,
        strict=False,
    ):
        if expected.name != actual.name:
            raise ValueError(
                "Feature schema mismatch: column name differs. "
                f"Expected '{expected.name}', got '{actual.name}'."
            )

        if expected.dataType != actual.dataType:
            raise ValueError(
                "Feature schema mismatch: data type differs for "
                f"column '{expected.name}'. "
                f"Expected '{expected.dataType.simpleString()}', "
                f"got '{actual.dataType.simpleString()}'."
            )


def validate_feature_values(df: DataFrame) -> None:
    """Reject null or non-finite values in numeric feature columns."""
    numeric_fields = [
        field for field in FEATURE_SCHEMA.fields if field.dataType in (DoubleType(), IntegerType())
    ]

    invalid_condition = None
    for field in numeric_fields:
        condition = F.col(field.name).isNull()
        if isinstance(field.dataType, DoubleType):
            condition = condition | F.isnan(field.name)
        invalid_condition = (
            condition if invalid_condition is None else invalid_condition | condition
        )

    if df.filter(invalid_condition).limit(1).count() > 0:
        raise ValueError("Feature dataset contains null or non-finite numeric values.")
