from dataclasses import dataclass
from math import isclose

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from ml.contracts.features import MODEL_FEATURE_COLUMNS
from ml.data.constants import VALID_BINARY_LABELS

TARGET_COLUMN = "is_fraud"
IDENTIFIER_COLUMNS = (
    "transaction_id",
    "origin_account_id",
    "destination_account_id",
    "timestamp",
)
METADATA_COLUMNS = (*IDENTIFIER_COLUMNS, "is_flagged_fraud")
MODEL_DATASET_COLUMNS = ("transaction_id", *MODEL_FEATURE_COLUMNS, TARGET_COLUMN)

__all__ = [
    "IDENTIFIER_COLUMNS",
    "METADATA_COLUMNS",
    "MODEL_DATASET_COLUMNS",
    "MODEL_FEATURE_COLUMNS",
    "SplitValidationResult",
    "TARGET_COLUMN",
    "chronological_split",
    "project_model_dataset",
    "validate_split_datasets",
    "validate_split_ratios",
]


@dataclass(frozen=True)
class SplitValidationResult:
    total_row_count: int
    train_row_count: int
    validation_row_count: int
    test_row_count: int
    invalid_target_count: int
    duplicate_transaction_id_count: int
    time_ranges: dict[str, dict[str, str | None]] | None = None

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
    *,
    strategy: str = "exact",
    quantile_relative_error: float = 0.01,
) -> dict[str, DataFrame]:
    """Split transactions chronologically and deterministically.

    ``exact`` preserves exact row-count boundaries with a global ordering window.  It is the
    compatibility/default mode because it is the only mode that guarantees the requested ratios
    for every input.  ``time_boundary`` uses distributed timestamp quantiles and avoids the global
    row-number window for normal data; it falls back to ``exact`` when duplicate timestamps would
    create an empty partition.
    """
    validate_split_ratios(train_ratio, validation_ratio, test_ratio)
    if strategy not in {"exact", "time_boundary"}:
        raise ValueError("strategy must be either 'exact' or 'time_boundary'")
    if not 0 <= quantile_relative_error <= 1:
        raise ValueError("quantile_relative_error must be between 0 and 1")

    required_columns = {"transaction_id", "timestamp"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing split columns: {sorted(missing_columns)}")

    if strategy == "time_boundary":
        return _chronological_split_by_time_boundaries(
            df,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            quantile_relative_error=quantile_relative_error,
        )

    row_count = df.count()
    if row_count < 3:
        raise ValueError("At least three rows are required for a three-way split.")

    return _chronological_split_exact(
        df,
        row_count=row_count,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )


def _chronological_split_exact(
    df: DataFrame,
    *,
    row_count: int,
    train_ratio: float,
    validation_ratio: float,
) -> dict[str, DataFrame]:
    """Use a global order only when exact row boundaries are required."""

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


def _chronological_split_by_time_boundaries(
    df: DataFrame,
    *,
    train_ratio: float,
    validation_ratio: float,
    quantile_relative_error: float,
) -> dict[str, DataFrame]:
    """Split by distributed timestamp quantiles, with a correctness fallback."""

    quantiles = df.select(
        F.col("timestamp").cast("double").alias("_timestamp_epoch")
    ).approxQuantile(
        "_timestamp_epoch",
        [train_ratio, train_ratio + validation_ratio],
        relativeError=quantile_relative_error,
    )
    if len(quantiles) != 2 or quantiles[0] >= quantiles[1]:
        return _chronological_split_exact(
            df,
            row_count=df.count(),
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )

    train_cutoff, validation_cutoff = quantiles
    splits = {
        "train": df.filter(F.col("timestamp").cast("double") <= train_cutoff),
        "validation": df.filter(
            (F.col("timestamp").cast("double") > train_cutoff)
            & (F.col("timestamp").cast("double") <= validation_cutoff)
        ),
        "test": df.filter(F.col("timestamp").cast("double") > validation_cutoff),
    }
    if any(split.limit(1).count() == 0 for split in splits.values()):
        return _chronological_split_exact(
            df,
            row_count=df.count(),
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )
    return splits


def project_model_dataset(df: DataFrame) -> DataFrame:
    """Project an audit feature frame to the model-ready dataset contract.

    ``transaction_id`` is retained solely for split integrity checks and ``is_fraud`` remains the
    target. Neither is passed to the model preprocessor. Post-event, historical, account, and
    proxy-label fields are intentionally dropped from the persisted training splits.
    """

    missing_columns = sorted(set(MODEL_DATASET_COLUMNS).difference(df.columns))
    if missing_columns:
        raise ValueError(f"Model dataset is missing required columns: {missing_columns}")
    return df.select(*MODEL_DATASET_COLUMNS)


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

    tagged_splits = [
        frame.select(
            "transaction_id",
            "timestamp",
            TARGET_COLUMN,
            F.lit(name).alias("_split_name"),
        )
        for name, frame in splits.items()
    ]
    split_stats = (
        tagged_splits[0]
        .unionByName(tagged_splits[1])
        .unionByName(tagged_splits[2])
        .groupBy("_split_name")
        .agg(
            F.count(F.lit(1)).alias("row_count"),
            F.sum(F.when(~F.col(TARGET_COLUMN).isin(*VALID_BINARY_LABELS), 1).otherwise(0)).alias(
                "invalid_target_count"
            ),
            F.min("timestamp").alias("min_timestamp"),
            F.max("timestamp").alias("max_timestamp"),
        )
        .collect()
    )
    stats_by_name = {row["_split_name"]: row for row in split_stats}
    row_counts = {name: int(stats_by_name[name]["row_count"] or 0) for name in splits}
    invalid_target_count = sum(
        int(stats_by_name[name]["invalid_target_count"] or 0) for name in splits
    )

    combined_ids = (
        train_df.select("transaction_id")
        .unionByName(validation_df.select("transaction_id"))
        .unionByName(test_df.select("transaction_id"))
    )
    duplicate_transaction_id_count = (
        combined_ids.groupBy("transaction_id").count().filter(F.col("count") > 1).count()
    )

    train_max = stats_by_name["train"]["max_timestamp"]
    validation_min = stats_by_name["validation"]["min_timestamp"]
    validation_max = stats_by_name["validation"]["max_timestamp"]
    test_min = stats_by_name["test"]["min_timestamp"]

    if train_max > validation_min or validation_max > test_min:
        raise ValueError("Split timestamp boundaries are not chronological.")

    return SplitValidationResult(
        total_row_count=all_df.count(),
        train_row_count=row_counts["train"],
        validation_row_count=row_counts["validation"],
        test_row_count=row_counts["test"],
        invalid_target_count=invalid_target_count,
        duplicate_transaction_id_count=duplicate_transaction_id_count,
        time_ranges={
            name: {
                "min": (
                    str(stats_by_name[name]["min_timestamp"])
                    if stats_by_name[name]["min_timestamp"] is not None
                    else None
                ),
                "max": (
                    str(stats_by_name[name]["max_timestamp"])
                    if stats_by_name[name]["max_timestamp"] is not None
                    else None
                ),
            }
            for name in splits
        },
    )
