"""Reproducible Sprint 2 model-training pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
from pyspark.sql import SparkSession

from app.core.logging import setup_logging
from ml.evaluation.selection import (
    FinalTestResult,
    ProductionCandidate,
    evaluate_production_candidate_on_test,
    select_production_candidate,
)
from ml.evaluation.threshold import analyze_thresholds, select_best_threshold
from ml.training.baseline import train_logistic_regression_baseline
from ml.training.config import ModelConfig, load_model_config
from ml.training.dataset import load_training_dataset
from ml.training.models import train_random_forest, train_xgboost
from ml.training.preprocessing import spark_frame_to_pandas

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingPipelineSummary:
    """Stable summary returned by one training pipeline run."""

    train_row_count: int
    validation_row_count: int
    test_row_count: int
    feature_count: int
    model_names: tuple[str, ...]
    production_model: str
    production_threshold: float
    evaluation_report: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_training_pipeline(
    spark: SparkSession,
    config: ModelConfig,
) -> TrainingPipelineSummary:
    """Train, select, and evaluate models from the configured feature splits."""

    LOGGER.info(
        "Starting model training pipeline: features_path=%s, target=%s, seed=%s",
        config.features_path,
        config.target_column,
        config.random_seed,
    )
    bundle = load_training_dataset(spark, config.features_path)
    feature_columns = bundle.summary.feature_columns
    LOGGER.info(
        "Training dataset contract loaded: train=%s, validation=%s, test=%s, features=%s",
        bundle.summary.split_summaries["train"].row_count,
        bundle.summary.split_summaries["validation"].row_count,
        bundle.summary.split_summaries["test"].row_count,
        len(feature_columns),
    )
    train_frame = spark_frame_to_pandas(bundle.train, (*feature_columns, config.target_column))
    validation_frame = spark_frame_to_pandas(
        bundle.validation, (*feature_columns, config.target_column)
    )
    test_frame = spark_frame_to_pandas(bundle.test, (*feature_columns, config.target_column))

    LOGGER.info(
        "Feature frames materialized: train=%s, validation=%s, test=%s",
        len(train_frame),
        len(validation_frame),
        len(test_frame),
    )
    LOGGER.info("Training models started")
    models = {
        "logistic_regression": train_logistic_regression_baseline(
            train_frame,
            feature_columns=feature_columns,
            config=config,
        ),
        "random_forest": train_random_forest(
            train_frame,
            feature_columns=feature_columns,
            config=config,
        ),
        "xgboost": train_xgboost(
            train_frame,
            feature_columns=feature_columns,
            config=config,
        ),
    }
    LOGGER.info("Training models completed: models=%s", tuple(models))
    validation_target = validation_frame[config.target_column]
    validation_predictions = {
        name: (validation_target, model.predict_proba(validation_frame)[:, 1])
        for name, model in models.items()
    }
    candidate = select_production_candidate(
        validation_predictions,
        thresholds=config.thresholds,
        primary_metric=config.primary_metric,
    )
    LOGGER.info(
        "Production candidate selected: model=%s, threshold=%s, metric=%s, score=%s",
        candidate.model_name,
        candidate.threshold,
        candidate.selection_metric,
        candidate.validation_score,
    )
    validation_report = _build_validation_report(
        validation_predictions,
        config,
    )

    test_target = test_frame[config.target_column]
    selected_model = models[candidate.model_name]
    final_result = evaluate_production_candidate_on_test(
        candidate,
        test_target,
        selected_model.predict_proba(test_frame)[:, 1],
    )
    LOGGER.info(
        "Final test evaluation completed: model=%s, threshold=%s, pr_auc=%.6f, f1=%.6f",
        candidate.model_name,
        candidate.threshold,
        final_result.metrics.pr_auc,
        final_result.metrics.f1,
    )
    report = _build_report(
        config=config,
        bundle_summary=bundle.summary.as_dict(),
        validation_report=validation_report,
        candidate=candidate,
        final_result=final_result,
    )
    _save_artifacts(models, config)
    LOGGER.info("Model artifacts saved: path=%s", config.model_directory)
    _write_json(config.evaluation_report, report)
    LOGGER.info("Evaluation report saved: path=%s", config.evaluation_report)
    summary = TrainingPipelineSummary(
        train_row_count=len(train_frame),
        validation_row_count=len(validation_frame),
        test_row_count=len(test_frame),
        feature_count=len(feature_columns),
        model_names=tuple(models),
        production_model=candidate.model_name,
        production_threshold=candidate.threshold,
        evaluation_report=str(config.evaluation_report),
    )
    LOGGER.info("Model training completed: %s", summary)
    return summary


def _build_validation_report(
    validation_predictions: dict[str, tuple[Any, Any]], config: ModelConfig
) -> dict[str, Any]:
    report = {}
    for model_name, (target, probability) in validation_predictions.items():
        evaluations = analyze_thresholds(target, probability, config.thresholds)
        selected = select_best_threshold(evaluations)
        report[model_name] = {
            "selected_threshold": selected.metrics.threshold,
            "metrics": selected.metrics.as_dict(),
            "expected_cost": selected.expected_cost,
        }
    return report


def _build_report(
    *,
    config: ModelConfig,
    bundle_summary: dict[str, Any],
    validation_report: dict[str, Any],
    candidate: ProductionCandidate,
    final_result: FinalTestResult,
) -> dict[str, Any]:
    return {
        "config": {
            "target_column": config.target_column,
            "random_seed": config.random_seed,
            "primary_metric": config.primary_metric,
            "thresholds": list(config.thresholds),
        },
        "dataset": bundle_summary,
        "validation": validation_report,
        "candidate": candidate.as_dict(),
        "test": final_result.metrics.as_dict(),
    }


def _save_artifacts(models: dict[str, Any], config: ModelConfig) -> None:
    config.model_directory.mkdir(parents=True, exist_ok=True)
    for model_name, model in models.items():
        joblib.dump(model, config.model_directory / f"{model_name}.joblib")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/model.yaml",
        help="Path to the model configuration YAML file",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = _parse_args()
    config = load_model_config(args.config)
    spark = SparkSession.builder.appName("TransactionRiskML-ModelTraining").getOrCreate()
    try:
        run_training_pipeline(spark, config)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
