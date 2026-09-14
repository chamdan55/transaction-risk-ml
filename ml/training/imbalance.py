"""Class-imbalance analysis for Sprint 2 model training."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class ClassImbalanceError(ValueError):
    """Raised when the target cannot be used for imbalance handling."""


@dataclass(frozen=True)
class TargetDistribution:
    """Binary target distribution summary."""

    total_count: int
    positive_count: int
    negative_count: int

    @property
    def positive_rate(self) -> float:
        """Return the fraction of positive examples."""

        if self.total_count == 0:
            return 0.0
        return self.positive_count / self.total_count


def summarize_target(target: pd.Series) -> TargetDistribution:
    """Summarize and validate a pandas binary target series."""

    if target.isna().any():
        raise ClassImbalanceError("Target contains null values")
    invalid_values = set(target.unique()).difference({0, 1})
    if invalid_values:
        raise ClassImbalanceError(
            f"Target must contain only 0 and 1; found {sorted(invalid_values)}"
        )

    negative_count = int((target == 0).sum())
    positive_count = int((target == 1).sum())
    return _distribution_from_counts(positive_count, negative_count)


def summarize_spark_target(dataframe: DataFrame, target_column: str) -> TargetDistribution:
    """Summarize a Spark binary target without collecting target rows."""

    if target_column not in dataframe.columns:
        raise ClassImbalanceError(f"Target column not found: {target_column}")
    summary = dataframe.agg(
        F.count(F.lit(1)).alias("total_count"),
        F.sum(F.when(F.col(target_column) == 1, 1).otherwise(0)).alias("positive_count"),
        F.sum(F.when(F.col(target_column) == 0, 1).otherwise(0)).alias("negative_count"),
        F.sum(F.when(F.col(target_column).isNull(), 1).otherwise(0)).alias("null_count"),
    ).first()
    null_count = int(summary["null_count"] or 0)
    positive_count = int(summary["positive_count"] or 0)
    negative_count = int(summary["negative_count"] or 0)
    total_count = int(summary["total_count"] or 0)
    invalid_count = total_count - null_count - positive_count - negative_count
    if null_count or invalid_count:
        raise ClassImbalanceError(
            f"Target contains invalid values: null_count={null_count}, "
            f"invalid_count={invalid_count}"
        )
    return _distribution_from_counts(positive_count, negative_count)


def balanced_class_weight(distribution: TargetDistribution) -> dict[int, float]:
    """Return sklearn-compatible balanced weights for classes 0 and 1."""

    if distribution.positive_count == 0 or distribution.negative_count == 0:
        raise ClassImbalanceError(
            "Both positive and negative classes are required for class weights"
        )
    total = distribution.total_count
    return {
        0: total / (2 * distribution.negative_count),
        1: total / (2 * distribution.positive_count),
    }


def scale_pos_weight(distribution: TargetDistribution) -> float:
    """Return the XGBoost positive-class scaling ratio."""

    if distribution.positive_count == 0:
        raise ClassImbalanceError("Positive class is required for scale_pos_weight")
    return distribution.negative_count / distribution.positive_count


def _distribution_from_counts(positive_count: int, negative_count: int) -> TargetDistribution:
    total_count = positive_count + negative_count
    if total_count == 0:
        raise ClassImbalanceError("Target must contain at least one row")
    return TargetDistribution(
        total_count=total_count,
        positive_count=positive_count,
        negative_count=negative_count,
    )
