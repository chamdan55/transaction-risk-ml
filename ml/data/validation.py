from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ml.data.constants import VALID_TRANSACTION_TYPES
from ml.data.domain_rules import (
    build_destination_balance_inconsistency_condition,
    build_negative_balance_condition,
    build_origin_balance_inconsistency_condition,
)
from ml.data.schema import CANONICAL_TRANSACTION_SCHEMA


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    row_count: int
    null_count: int
    invalid_amount_count: int
    invalid_fraud_label_count: int
    invalid_flagged_fraud_count: int
    invalid_step_count: int


@dataclass(frozen=True)
class CanonicalValidationResult:
    is_valid: bool
    row_count: int
    missing_column_count: int
    null_count: int
    invalid_transaction_id_count: int
    duplicate_transaction_id_count: int
    invalid_amount_count: int
    invalid_fraud_label_count: int
    invalid_flagged_fraud_count: int
    invalid_transaction_type_count: int


@dataclass(frozen=True)
class DomainValidationResult:
    negative_balance_count: int
    origin_balance_mismatch_count: int
    destination_balance_mismatch_count: int


def validate_paysim(df: DataFrame) -> ValidationResult:
    required_columns = {
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    row_count = df.count()

    null_condition = F.lit(False)

    for column in df.columns:
        null_condition = null_condition | F.col(column).isNull()

    null_count = df.filter(null_condition).count()

    invalid_amount_count = df.filter(F.col("amount") < 0).count()

    invalid_fraud_label_count = df.filter(~F.col("isFraud").isin(0, 1)).count()

    invalid_flagged_fraud_count = df.filter(~F.col("isFlaggedFraud").isin(0, 1)).count()

    invalid_step_count = df.filter(F.col("step") < 0).count()

    is_valid = all(
        [
            row_count > 0,
            null_count == 0,
            invalid_amount_count == 0,
            invalid_fraud_label_count == 0,
            invalid_flagged_fraud_count == 0,
            invalid_step_count == 0,
        ]
    )

    return ValidationResult(
        is_valid=is_valid,
        row_count=row_count,
        null_count=null_count,
        invalid_amount_count=invalid_amount_count,
        invalid_fraud_label_count=invalid_fraud_label_count,
        invalid_flagged_fraud_count=invalid_flagged_fraud_count,
        invalid_step_count=invalid_step_count,
    )


def validate_canonical_schema(df: DataFrame) -> None:
    expected_fields = CANONICAL_TRANSACTION_SCHEMA.fields
    actual_fields = df.schema.fields

    if len(actual_fields) != len(expected_fields):
        raise ValueError(
            "Canonical schema mismatch: "
            f"expected {len(expected_fields)} fields, "
            f"got {len(actual_fields)}."
        )

    for expected, actual in zip(expected_fields, actual_fields, strict=False):
        if expected.name != actual.name:
            raise ValueError(
                "Canonical schema mismatch: column name differs. "
                f"Expected '{expected.name}', got '{actual.name}'."
            )

        if expected.dataType != actual.dataType:
            raise ValueError(
                "Canonical schema mismatch: data type differs for "
                f"column '{expected.name}'. "
                f"Expected '{expected.dataType.simpleString()}', "
                f"got '{actual.dataType.simpleString()}'."
            )


def validate_canonical_data(df: DataFrame) -> CanonicalValidationResult:
    validate_canonical_schema(df)

    row_count = df.count()

    required_columns = [
        "transaction_id",
        "timestamp",
        "transaction_type",
        "origin_account_id",
        "destination_account_id",
        "amount",
        "is_fraud",
        "is_flagged_fraud",
    ]

    missing_column_count = sum(column not in df.columns for column in required_columns)

    null_condition = None

    for column in required_columns:
        condition = F.col(column).isNull()

        null_condition = condition if null_condition is None else null_condition | condition

    null_count = df.filter(null_condition).count()

    invalid_transaction_id_count = df.filter(
        F.col("transaction_id").isNull() | (F.length("transaction_id") != 64)
    ).count()

    duplicate_transaction_id_count = (
        df.groupBy("transaction_id").count().filter(F.col("count") > 1).count()
    )

    invalid_amount_count = df.filter(F.col("amount").isNull() | (F.col("amount") < 0)).count()

    invalid_fraud_label_count = df.filter(~F.col("is_fraud").isin(0, 1)).count()

    invalid_flagged_fraud_count = df.filter(~F.col("is_flagged_fraud").isin(0, 1)).count()

    invalid_transaction_type_count = df.filter(
        ~F.col("transaction_type").isin(*VALID_TRANSACTION_TYPES)
    ).count()

    is_valid = all(
        [
            missing_column_count == 0,
            null_count == 0,
            invalid_transaction_id_count == 0,
            duplicate_transaction_id_count == 0,
            invalid_amount_count == 0,
            invalid_fraud_label_count == 0,
            invalid_flagged_fraud_count == 0,
            invalid_transaction_type_count == 0,
        ]
    )

    return CanonicalValidationResult(
        is_valid=is_valid,
        row_count=row_count,
        missing_column_count=missing_column_count,
        null_count=null_count,
        invalid_transaction_id_count=invalid_transaction_id_count,
        duplicate_transaction_id_count=duplicate_transaction_id_count,
        invalid_amount_count=invalid_amount_count,
        invalid_fraud_label_count=invalid_fraud_label_count,
        invalid_flagged_fraud_count=invalid_flagged_fraud_count,
        invalid_transaction_type_count=invalid_transaction_type_count,
    )


def validate_transaction_domain(
    df: DataFrame,
) -> DomainValidationResult:
    validate_canonical_schema(df)

    negative_balance_count = df.filter(build_negative_balance_condition()).count()

    inconsistent_origin_balance_count = df.filter(
        build_origin_balance_inconsistency_condition()
    ).count()

    inconsistent_destination_balance_count = df.filter(
        build_destination_balance_inconsistency_condition()
    ).count()

    return DomainValidationResult(
        negative_balance_count=negative_balance_count,
        origin_balance_mismatch_count=inconsistent_origin_balance_count,
        destination_balance_mismatch_count=inconsistent_destination_balance_count,
    )
