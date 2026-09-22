from __future__ import annotations

import importlib.util
import json
from dataclasses import replace

import pytest

from ml.data.split import MODEL_FEATURE_COLUMNS
from ml.training.config import load_model_config
from pipelines.train_models import run_training_pipeline

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("xgboost") is None,
    reason="xgboost is required for the training pipeline integration test",
)


def _write_feature_splits(spark, root):
    columns = ["transaction_id", *MODEL_FEATURE_COLUMNS, "is_fraud"]
    for split_index, split_name in enumerate(("train", "validation", "test")):
        rows = []
        for row_index in range(12):
            transaction_id = split_index * 1000 + row_index
            row = [transaction_id]
            row.extend(
                float(row_index + feature_index)
                for feature_index, _ in enumerate(MODEL_FEATURE_COLUMNS)
            )
            row.append(row_index % 2)
            rows.append(tuple(row))
        spark.createDataFrame(rows, columns).write.mode("overwrite").parquet(str(root / split_name))


def test_training_pipeline_writes_reproducible_artifacts(spark, tmp_path):
    features_path = tmp_path / "features"
    _write_feature_splits(spark, features_path)
    config = load_model_config("configs/model.yaml")
    config = replace(
        config,
        features_path=features_path,
        model_directory=tmp_path / "models",
        evaluation_report=tmp_path / "evaluation_report.json",
    )
    config.model_params["random_forest"]["n_jobs"] = 1
    config.model_params["random_forest"]["n_estimators"] = 5
    config.model_params["xgboost"].update({"n_jobs": 1, "n_estimators": 5, "max_depth": 2})

    first_summary = run_training_pipeline(spark, config)
    first_report = config.evaluation_report.read_text(encoding="utf-8")
    second_summary = run_training_pipeline(spark, config)
    second_report = config.evaluation_report.read_text(encoding="utf-8")

    assert first_summary == second_summary
    assert first_report == second_report
    assert first_summary.model_names == (
        "logistic_regression",
        "random_forest",
        "xgboost",
    )
    assert config.evaluation_report.exists()
    report = json.loads(first_report)
    assert set(report["validation"]) == set(first_summary.model_names)
    assert all(
        {"status", "selected_threshold", "threshold_evaluations"}.issubset(model_report)
        for model_report in report["validation"].values()
    )
    assert all(
        (config.model_directory / f"{model_name}.joblib").exists()
        for model_name in first_summary.model_names
    )
