"""Dataset contract for Sprint 2 model training.

The contract intentionally consumes only the chronological feature splits
produced by Sprint 1. It validates the shape of each split and checks that
transaction identifiers cannot leak between splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ml.data.split import MODEL_FEATURE_COLUMNS, TARGET_COLUMN

TRANSACTION_ID_COLUMN = "transaction_id"
SPLIT_NAMES = ("train", "validation", "test")


class TrainingDatasetContractError(ValueError):
    """Raised when a feature split violates the training dataset contract."""


@dataclass(frozen=True)
class SplitSummary:
    """Summary statistics for one feature split."""

    row_count: int
    positive_target_count: int
    negative_target_count: int


@dataclass(frozen=True)
class TrainingDatasetSummary:
    """Summary statistics for the complete training dataset bundle."""

    split_summaries: dict[str, SplitSummary]
    feature_columns: tuple[str, ...]
    target_column: str
    duplicate_transaction_id_counts: dict[str, int]
    overlapping_transaction_id_counts: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation."""

        return {
            "split_summaries": {
                name: {
                    "row_count": summary.row_count,
                    "positive_target_count": summary.positive_target_count,
                    "negative_target_count": summary.negative_target_count,
                }
                for name, summary in self.split_summaries.items()
            },
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "duplicate_transaction_id_counts": self.duplicate_transaction_id_counts,
            "overlapping_transaction_id_counts": self.overlapping_transaction_id_counts,
        }


@dataclass(frozen=True)
class TrainingDatasetBundle:
    """Validated Spark DataFrames and their contract summary."""

    train: DataFrame
    validation: DataFrame
    test: DataFrame
    summary: TrainingDatasetSummary

    def by_name(self, split_name: str) -> DataFrame:
        """Return a split by its canonical name."""

        if split_name not in SPLIT_NAMES:
            raise KeyError(f"Unknown split: {split_name}")
        return getattr(self, split_name)


def load_training_dataset(
    spark: SparkSession,
    features_path: str | Path,
    *,
    target_column: str = TARGET_COLUMN,
    model_feature_columns: tuple[str, ...] = tuple(MODEL_FEATURE_COLUMNS),
) -> TrainingDatasetBundle:
    """Read and validate the Sprint 1 feature splits from Parquet.

    The loader only accepts the canonical ``train``, ``validation``, and
    ``test`` directories below ``features_path``. It does not modify or
    rewrite the source data.
    """

    root = Path(features_path)
    splits: dict[str, DataFrame] = {}

    for split_name in SPLIT_NAMES:
        split_path = root / split_name
        try:
            dataframe = spark.read.parquet(str(split_path))
        except Exception as exc:
            raise TrainingDatasetContractError(
                f"Unable to read {split_name} feature split from {split_path}"
            ) from exc
        _validate_split_schema(
            dataframe,
            split_name=split_name,
            target_column=target_column,
            model_feature_columns=model_feature_columns,
        )
        splits[split_name] = dataframe

    duplicate_counts = {
        split_name: _count_duplicate_transaction_ids(dataframe)
        for split_name, dataframe in splits.items()
    }
    duplicate_splits = {
        split_name: count for split_name, count in duplicate_counts.items() if count > 0
    }
    if duplicate_splits:
        raise TrainingDatasetContractError(
            f"Duplicate transaction_id values found: {duplicate_splits}"
        )

    overlap_counts = _count_split_overlaps(splits)
    overlapping_splits = {pair: count for pair, count in overlap_counts.items() if count > 0}
    if overlapping_splits:
        raise TrainingDatasetContractError(
            f"transaction_id values overlap between splits: {overlapping_splits}"
        )

    split_summaries = {
        split_name: _build_split_summary(dataframe, target_column)
        for split_name, dataframe in splits.items()
    }
    summary = TrainingDatasetSummary(
        split_summaries=split_summaries,
        feature_columns=tuple(model_feature_columns),
        target_column=target_column,
        duplicate_transaction_id_counts=duplicate_counts,
        overlapping_transaction_id_counts=overlap_counts,
    )
    return TrainingDatasetBundle(
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
        summary=summary,
    )


def _validate_split_schema(
    dataframe: DataFrame,
    *,
    split_name: str,
    target_column: str,
    model_feature_columns: tuple[str, ...],
) -> None:
    required_columns = {
        TRANSACTION_ID_COLUMN,
        target_column,
        *model_feature_columns,
    }
    missing_columns = sorted(required_columns.difference(dataframe.columns))
    if missing_columns:
        raise TrainingDatasetContractError(
            f"{split_name} split is missing required columns: {missing_columns}"
        )

    if len(set(model_feature_columns)) != len(model_feature_columns):
        raise TrainingDatasetContractError("model_feature_columns contains duplicates")

    forbidden_columns = {
        TRANSACTION_ID_COLUMN,
        target_column,
        "is_flagged_fraud",
        "timestamp",
        "origin_account_id",
        "destination_account_id",
    }
    forbidden_features = forbidden_columns.intersection(model_feature_columns)
    if forbidden_features:
        raise TrainingDatasetContractError(
            f"Forbidden columns included as model features: {sorted(forbidden_features)}"
        )


def _count_duplicate_transaction_ids(dataframe: DataFrame) -> int:
    duplicate_rows = dataframe.groupBy(TRANSACTION_ID_COLUMN).count().where(F.col("count") > 1)
    return duplicate_rows.count()


def _count_split_overlaps(splits: dict[str, DataFrame]) -> dict[str, int]:
    overlap_counts: dict[str, int] = {}
    for left_index, left_name in enumerate(SPLIT_NAMES):
        for right_name in SPLIT_NAMES[left_index + 1 :]:
            left_ids = splits[left_name].select(TRANSACTION_ID_COLUMN).distinct()
            right_ids = splits[right_name].select(TRANSACTION_ID_COLUMN).distinct()
            overlap_counts[f"{left_name}__{right_name}"] = left_ids.join(
                right_ids,
                on=TRANSACTION_ID_COLUMN,
                how="inner",
            ).count()
    return overlap_counts


def _build_split_summary(dataframe: DataFrame, target_column: str) -> SplitSummary:
    counts = dataframe.agg(
        F.count(F.lit(1)).alias("row_count"),
        F.sum(F.when(F.col(target_column) == 1, 1).otherwise(0)).alias("positive_target_count"),
        F.sum(F.when(F.col(target_column) == 0, 1).otherwise(0)).alias("negative_target_count"),
    ).first()
    return SplitSummary(
        row_count=int(counts["row_count"] or 0),
        positive_target_count=int(counts["positive_target_count"] or 0),
        negative_target_count=int(counts["negative_target_count"] or 0),
    )
