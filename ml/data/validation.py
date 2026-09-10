from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    row_count: int
    null_count: int
    invalid_amount_count: int
    invalid_fraud_label_count: int
    invalid_flagged_fraud_count: int
    invalid_step_count: int


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
