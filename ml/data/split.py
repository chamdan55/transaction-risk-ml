from dataclasses import dataclass
from math import isclose

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from ml.data.constants import VALID_BINARY_LABELS
from ml.features.schema import FEATURE_SCHEMA

TARGET_COLUMN = "is_fraud"
IDENTIFIER_COLUMNS = (
    "transaction_id",
    "origin_account_id",
    "destination_account_id",
    "timestamp",
)
METADATA_COLUMNS = (*IDENTIFIER_COLUMNS, "is_flagged_fraud")
MODEL_FEATURE_COLUMNS = tuple(
    column for column in FEATURE_SCHEMA.names if column not in (*METADATA_COLUMNS, TARGET_COLUMN)
)


@dataclass(frozen=True)
class SplitValidationResult:
    total_row_count: int
    train_row_count: int
    validation_row_count: int
    test_row_count: int
    invalid_target_count: int
    duplicate_transaction_id_count: int

    @property
    def is_valid(self) -> bool:
        return (
            self.train_row_count + self.validation_row_count + self.test_row_count
            == self.total_row_count
            and self.invalid_target_count == 0
            and self.duplicate_transaction_id_count == 0
        )


def validate_split_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> None:
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio <= 0 or ratio >= 1 for ratio in ratios):
        raise ValueError("Split ratios must be greater than 0 and less than 1.")

    if not isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError("Train, validation, and test ratios must sum to 1.0.")


def chronological_split(
    df: DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> dict[str, DataFrame]:
    """Split transactions chronologically and deterministically."""
    validate_split_ratios(train_ratio, validation_ratio, test_ratio)

    required_columns = {"transaction_id", "timestamp"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing split columns: {sorted(missing_columns)}")

    row_count = df.count()
    if row_count < 3:
        raise ValueError("At least three rows are required for a three-way split.")

    train_end = int(row_count * train_ratio)
    validation_end = train_end + int(row_count * validation_ratio)

    if train_end == 0 or validation_end == train_end or validation_end >= row_count:
        raise ValueError("Split ratios produce an empty dataset partition.")

    ordering_window = Window.orderBy(
        F.col("timestamp"),
        F.col("transaction_id"),
    )
    numbered_df = df.withColumn(
        "_split_row_number",
        F.row_number().over(ordering_window),
    )

    return {
        "train": numbered_df.filter(F.col("_split_row_number") <= train_end).drop(
            "_split_row_number"
        ),
        "validation": numbered_df.filter(
            (F.col("_split_row_number") > train_end)
            & (F.col("_split_row_number") <= validation_end)
        ).drop("_split_row_number"),
        "test": numbered_df.filter(F.col("_split_row_number") > validation_end).drop(
            "_split_row_number"
        ),
    }


def validate_split_datasets(
    all_df: DataFrame,
    splits: dict[str, DataFrame],
) -> SplitValidationResult:
    """Validate row counts, target values, IDs, schemas, and time boundaries."""
    expected_split_names = {"train", "validation", "test"}
    if set(splits) != expected_split_names:
        raise ValueError(f"Expected splits: {sorted(expected_split_names)}")

    train_df = splits["train"]
    validation_df = splits["validation"]
    test_df = splits["test"]

    if not (train_df.schema == validation_df.schema == test_df.schema == all_df.schema):
        raise ValueError("Split schemas must match the all-feature dataset schema.")

    row_counts = {name: frame.count() for name, frame in splits.items()}
    invalid_target_count = sum(
        frame.filter(~F.col(TARGET_COLUMN).isin(*VALID_BINARY_LABELS)).count()
        for frame in splits.values()
    )

    combined_ids = (
        train_df.select("transaction_id")
        .unionByName(validation_df.select("transaction_id"))
        .unionByName(test_df.select("transaction_id"))
    )
    duplicate_transaction_id_count = (
        combined_ids.groupBy("transaction_id").count().filter(F.col("count") > 1).count()
    )

    train_max = train_df.select(F.max("timestamp")).first()[0]
    validation_min = validation_df.select(F.min("timestamp")).first()[0]
    validation_max = validation_df.select(F.max("timestamp")).first()[0]
    test_min = test_df.select(F.min("timestamp")).first()[0]

    if train_max > validation_min or validation_max > test_min:
        raise ValueError("Split timestamp boundaries are not chronological.")

    return SplitValidationResult(
        total_row_count=all_df.count(),
        train_row_count=row_counts["train"],
        validation_row_count=row_counts["validation"],
        test_row_count=row_counts["test"],
        invalid_target_count=invalid_target_count,
        duplicate_transaction_id_count=duplicate_transaction_id_count,
    )
