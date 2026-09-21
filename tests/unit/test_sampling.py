from pyspark.sql import SparkSession

from ml.training.sampling import (
    deterministically_sample_rows,
    retain_all_positives_and_sample_negatives,
)


def test_sampling_retains_all_positives_and_deterministically_caps_negatives(spark: SparkSession):
    dataframe = spark.createDataFrame(
        [(f"tx-{index}", 1 if index in {3, 8} else 0) for index in range(10)],
        ["transaction_id", "is_fraud"],
    )

    first, first_summary = retain_all_positives_and_sample_negatives(
        dataframe,
        target_column="is_fraud",
        max_rows=5,
        random_seed=42,
    )
    second, second_summary = retain_all_positives_and_sample_negatives(
        dataframe,
        target_column="is_fraud",
        max_rows=5,
        random_seed=42,
    )

    assert first_summary.source_row_count == 10
    assert first_summary.sampled_positive_count == 2
    assert first_summary.sampled_negative_count == 3
    assert first_summary.sampled_row_count == 5
    assert first_summary == second_summary
    assert {row.transaction_id for row in first.collect()} == {
        row.transaction_id for row in second.collect()
    }


def test_evaluation_sampling_does_not_force_all_positives_into_the_output(spark: SparkSession):
    dataframe = spark.createDataFrame(
        [(f"tx-{index}", 1 if index < 4 else 0) for index in range(20)],
        ["transaction_id", "is_fraud"],
    )

    _, summary = deterministically_sample_rows(
        dataframe,
        target_column="is_fraud",
        max_rows=5,
        random_seed=42,
    )

    assert summary.sampled_row_count == 5
    assert summary.sampled_positive_count <= summary.source_positive_count
