"""Deterministic, class-aware sampling before Spark data reaches pandas."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


@dataclass(frozen=True)
class SamplingSummary:
    """Source and retained class counts for one training split."""

    source_row_count: int
    sampled_row_count: int
    source_positive_count: int
    sampled_positive_count: int
    source_negative_count: int
    sampled_negative_count: int
    max_rows: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def summarize_rows(
    df: DataFrame,
    *,
    target_column: str,
    max_rows: int,
) -> SamplingSummary:
    """Summarize a split without materializing it in the driver."""

    if max_rows < 1:
        raise ValueError("max_rows must be at least 1")
    source_counts = _count_rows_by_target(df, target_column)
    return _build_summary(source_counts, source_counts, max_rows)


def retain_all_positives_and_sample_negatives(
    df: DataFrame,
    *,
    target_column: str,
    max_rows: int,
    random_seed: int,
) -> tuple[DataFrame, SamplingSummary]:
    """Keep every fraud row and deterministically cap only non-fraud rows.

    Scikit-learn training materializes data in driver memory. This operation
    keeps the rare positive class intact while selecting a stable subset of
    negatives by transaction ID before ``toPandas`` is called.
    """

    if max_rows < 1:
        raise ValueError("max_rows must be at least 1")
    if target_column not in df.columns or "transaction_id" not in df.columns:
        raise ValueError("Sampling requires target_column and transaction_id")

    counts = df.agg(
        F.count(F.lit(1)).alias("row_count"),
        F.sum(F.when(F.col(target_column) == 1, 1).otherwise(0)).alias("positive_count"),
    ).first()
    source_row_count = int(counts["row_count"] or 0)
    source_positive_count = int(counts["positive_count"] or 0)
    source_negative_count = source_row_count - source_positive_count

    if source_row_count <= max_rows:
        return df, SamplingSummary(
            source_row_count=source_row_count,
            sampled_row_count=source_row_count,
            source_positive_count=source_positive_count,
            sampled_positive_count=source_positive_count,
            source_negative_count=source_negative_count,
            sampled_negative_count=source_negative_count,
            max_rows=max_rows,
        )

    positive_df = df.filter(F.col(target_column) == 1)
    negative_limit = max(max_rows - source_positive_count, 0)
    if negative_limit == 0:
        sampled_df = positive_df
        sampled_negative_count = 0
    else:
        stable_order = F.sha2(
            F.concat_ws("||", F.col("transaction_id"), F.lit(str(random_seed))),
            256,
        )
        negative_df = df.filter(F.col(target_column) == 0).withColumn(
            "_sampling_rank",
            F.row_number().over(Window.orderBy(stable_order)),
        )
        selected_negative_df = negative_df.filter(F.col("_sampling_rank") <= negative_limit).drop(
            "_sampling_rank"
        )
        sampled_df = positive_df.unionByName(selected_negative_df)
        sampled_negative_count = negative_limit

    return sampled_df, SamplingSummary(
        source_row_count=source_row_count,
        sampled_row_count=source_positive_count + sampled_negative_count,
        source_positive_count=source_positive_count,
        sampled_positive_count=source_positive_count,
        source_negative_count=source_negative_count,
        sampled_negative_count=sampled_negative_count,
        max_rows=max_rows,
    )


def deterministically_sample_rows(
    df: DataFrame,
    *,
    target_column: str,
    max_rows: int,
    random_seed: int,
    strategy: str = "exact",
) -> tuple[DataFrame, SamplingSummary]:
    """Cap an evaluation split deterministically.

    ``exact`` uses a global hash ordering because callers explicitly request an exact row cap.
    ``hash`` uses a distributed hash predicate and is preferred when ``max_rows`` is a target
    rather than a hard business limit; its returned summary records the actual retained count.
    """

    if max_rows < 1:
        raise ValueError("max_rows must be at least 1")
    if strategy not in {"exact", "hash"}:
        raise ValueError("strategy must be either 'exact' or 'hash'")
    if target_column not in df.columns or "transaction_id" not in df.columns:
        raise ValueError("Sampling requires target_column and transaction_id")

    source_counts = _count_rows_by_target(df, target_column)
    if source_counts["row_count"] <= max_rows:
        return df, _build_summary(source_counts, source_counts, max_rows)

    stable_order = F.sha2(
        F.concat_ws("||", F.col("transaction_id"), F.lit(str(random_seed))),
        256,
    )
    if strategy == "hash":
        threshold = max(1, int(max_rows / source_counts["row_count"] * 1_000_000))
        sampled_df = (
            df.withColumn("_sampling_hash", F.pmod(F.xxhash64(stable_order), F.lit(1_000_000)))
            .filter(F.col("_sampling_hash") < threshold)
            .drop("_sampling_hash")
        )
        return sampled_df, _build_summary(
            source_counts,
            _count_rows_by_target(sampled_df, target_column),
            max_rows,
        )

    # Exact caps require a global ordering; this path is intentionally explicit and is reserved
    # for training caps where the retained row count is part of the reproducibility contract.
    sampled_df = (
        df.withColumn("_sampling_rank", F.row_number().over(Window.orderBy(stable_order)))
        .filter(F.col("_sampling_rank") <= max_rows)
        .drop("_sampling_rank")
    )
    return sampled_df, _build_summary(
        source_counts,
        _count_rows_by_target(sampled_df, target_column),
        max_rows,
    )


def _count_rows_by_target(df: DataFrame, target_column: str) -> dict[str, int]:
    counts = df.agg(
        F.count(F.lit(1)).alias("row_count"),
        F.sum(F.when(F.col(target_column) == 1, 1).otherwise(0)).alias("positive_count"),
    ).first()
    row_count = int(counts["row_count"] or 0)
    positive_count = int(counts["positive_count"] or 0)
    return {
        "row_count": row_count,
        "positive_count": positive_count,
        "negative_count": row_count - positive_count,
    }


def _build_summary(
    source_counts: dict[str, int],
    sampled_counts: dict[str, int],
    max_rows: int,
) -> SamplingSummary:
    return SamplingSummary(
        source_row_count=source_counts["row_count"],
        sampled_row_count=sampled_counts["row_count"],
        source_positive_count=source_counts["positive_count"],
        sampled_positive_count=sampled_counts["positive_count"],
        source_negative_count=source_counts["negative_count"],
        sampled_negative_count=sampled_counts["negative_count"],
        max_rows=max_rows,
    )
