from pathlib import Path

import pytest
import yaml
from pyspark.sql import SparkSession

from pipelines.run_pipeline import run_pipeline

RAW_HEADER = (
    "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,"
    "nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n"
)
RAW_ROWS = (
    "1,PAYMENT,100.0,C001,1000.0,900.0,M001,500.0,600.0,0,0\n"
    "2,TRANSFER,200.0,C001,900.0,700.0,C002,0.0,200.0,1,0\n"
    "3,CASH_IN,50.0,C002,0.0,50.0,C001,700.0,750.0,0,0\n"
    "4,DEBIT,25.0,C002,50.0,25.0,C003,0.0,0.0,0,0\n"
    "5,CASH_OUT,10.0,C001,700.0,690.0,C003,0.0,10.0,0,0\n"
    "6,PAYMENT,15.0,C001,690.0,675.0,M002,10.0,25.0,0,0\n"
)


def _write_pipeline_config(tmp_path: Path, raw_content: str) -> Path:
    raw_path = tmp_path / "paysim.csv"
    raw_path.write_text(raw_content, encoding="utf-8")

    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "raw_path": str(raw_path),
                    "processed_path": str(tmp_path / "processed"),
                },
                "time": {"base_timestamp": "2026-01-01T00:00:00"},
                "split": {
                    "train_ratio": 0.50,
                    "validation_ratio": 0.25,
                    "test_ratio": 0.25,
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _snapshot_outputs(spark: SparkSession, processed_path: Path) -> dict:
    paths = {
        "canonical": processed_path / "canonical",
        "train": processed_path / "features" / "train",
        "validation": processed_path / "features" / "validation",
        "test": processed_path / "features" / "test",
    }
    return {
        name: (
            spark.read.parquet(str(path)).schema,
            spark.read.parquet(str(path)).orderBy("transaction_id").collect(),
        )
        for name, path in paths.items()
    }


def test_run_pipeline_completes_from_raw_csv_to_split_parquet(
    spark: SparkSession,
    tmp_path: Path,
):
    config_path = _write_pipeline_config(tmp_path, RAW_HEADER + RAW_ROWS)

    summary = run_pipeline(config_path=config_path, spark=spark)

    assert summary.input_row_count == 6
    assert summary.canonical_row_count == 6
    assert summary.feature_row_count == 6
    assert summary.train_row_count == 3
    assert summary.validation_row_count == 1
    assert summary.test_row_count == 2
    assert summary.target_column == "is_fraud"

    output_root = Path(summary.feature_output_path)
    assert (output_root / "_SUCCESS").exists()
    assert (output_root / "train" / "_SUCCESS").exists()
    assert (output_root / "validation" / "_SUCCESS").exists()
    assert (output_root / "test" / "_SUCCESS").exists()


def test_run_pipeline_fails_before_writing_canonical_output_for_invalid_raw_data(
    spark: SparkSession,
    tmp_path: Path,
):
    invalid_rows = RAW_ROWS.replace("100.0,C001", "-100.0,C001", 1)
    config_path = _write_pipeline_config(tmp_path, RAW_HEADER + invalid_rows)

    with pytest.raises(ValueError, match="Raw dataset validation failed"):
        run_pipeline(config_path=config_path, spark=spark)

    assert not (tmp_path / "processed" / "canonical").exists()


def test_run_pipeline_is_idempotent_for_same_input(
    spark: SparkSession,
    tmp_path: Path,
):
    config_path = _write_pipeline_config(tmp_path, RAW_HEADER + RAW_ROWS)
    processed_path = tmp_path / "processed"

    first_summary = run_pipeline(config_path=config_path, spark=spark)
    first_outputs = _snapshot_outputs(spark, processed_path)

    second_summary = run_pipeline(config_path=config_path, spark=spark)
    second_outputs = _snapshot_outputs(spark, processed_path)

    assert first_summary == second_summary
    assert first_outputs == second_outputs
