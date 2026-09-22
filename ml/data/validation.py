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

    null_condition = F.lit(False)
    for column in df.columns:
        null_condition = null_condition | F.col(column).isNull()
    summary = df.agg(
        F.count(F.lit(1)).alias("row_count"),
        F.sum(F.when(null_condition, 1).otherwise(0)).alias("null_count"),
        F.sum(F.when(F.col("amount") < 0, 1).otherwise(0)).alias("invalid_amount_count"),
        F.sum(F.when(~F.col("isFraud").isin(0, 1), 1).otherwise(0)).alias(
            "invalid_fraud_label_count"
        ),
        F.sum(F.when(~F.col("isFlaggedFraud").isin(0, 1), 1).otherwise(0)).alias(
            "invalid_flagged_fraud_count"
        ),
        F.sum(F.when(F.col("step") < 0, 1).otherwise(0)).alias("invalid_step_count"),
    ).first()
    row_count = int(summary["row_count"] or 0)
    null_count = int(summary["null_count"] or 0)
    invalid_amount_count = int(summary["invalid_amount_count"] or 0)
    invalid_fraud_label_count = int(summary["invalid_fraud_label_count"] or 0)
    invalid_flagged_fraud_count = int(summary["invalid_flagged_fraud_count"] or 0)
    invalid_step_count = int(summary["invalid_step_count"] or 0)

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

    null_condition = F.lit(False)
    for column in required_columns:
        null_condition = null_condition | F.col(column).isNull()
    summary = df.agg(
        F.count(F.lit(1)).alias("row_count"),
        F.sum(F.when(null_condition, 1).otherwise(0)).alias("null_count"),
        F.sum(
            F.when(
                F.col("transaction_id").isNull() | (F.length("transaction_id") != 64),
                1,
            ).otherwise(0)
        ).alias("invalid_transaction_id_count"),
        F.sum(F.when(F.col("amount").isNull() | (F.col("amount") < 0), 1).otherwise(0)).alias(
            "invalid_amount_count"
        ),
        F.sum(F.when(~F.col("is_fraud").isin(0, 1), 1).otherwise(0)).alias(
            "invalid_fraud_label_count"
        ),
        F.sum(F.when(~F.col("is_flagged_fraud").isin(0, 1), 1).otherwise(0)).alias(
            "invalid_flagged_fraud_count"
        ),
        F.sum(
            F.when(~F.col("transaction_type").isin(*VALID_TRANSACTION_TYPES), 1).otherwise(0)
        ).alias("invalid_transaction_type_count"),
    ).first()
    row_count = int(summary["row_count"] or 0)
    null_count = int(summary["null_count"] or 0)
    invalid_transaction_id_count = int(summary["invalid_transaction_id_count"] or 0)
    invalid_amount_count = int(summary["invalid_amount_count"] or 0)
    invalid_fraud_label_count = int(summary["invalid_fraud_label_count"] or 0)
    invalid_flagged_fraud_count = int(summary["invalid_flagged_fraud_count"] or 0)
    invalid_transaction_type_count = int(summary["invalid_transaction_type_count"] or 0)
    duplicate_transaction_id_count = (
        df.groupBy("transaction_id").count().filter(F.col("count") > 1).count()
    )

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

    summary = df.agg(
        F.sum(F.when(build_negative_balance_condition(), 1).otherwise(0)).alias(
            "negative_balance_count"
        ),
        F.sum(F.when(build_origin_balance_inconsistency_condition(), 1).otherwise(0)).alias(
            "origin_balance_mismatch_count"
        ),
        F.sum(F.when(build_destination_balance_inconsistency_condition(), 1).otherwise(0)).alias(
            "destination_balance_mismatch_count"
        ),
    ).first()

    return DomainValidationResult(
        negative_balance_count=int(summary["negative_balance_count"] or 0),
        origin_balance_mismatch_count=int(summary["origin_balance_mismatch_count"] or 0),
        destination_balance_mismatch_count=int(summary["destination_balance_mismatch_count"] or 0),
    )
