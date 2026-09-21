"""Reproducible Sprint 2 training pipeline with Sprint 3 MLflow tracking."""

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
from ml.tracking.client import MlflowTrackingClient
from ml.tracking.config import TrackingConfig, load_tracking_config
from ml.tracking.logging import log_training_parameters, log_validation_metrics
from ml.tracking.metadata import build_run_metadata
from ml.tracking.registry import load_logged_model, log_and_register_model
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
    tracking_parent_run_id: str | None = None
    tracking_candidate_run_id: str | None = None
    registered_model_version: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_training_pipeline(
    spark: SparkSession,
    config: ModelConfig,
    *,
    tracking_config: TrackingConfig | None = None,
    model_config_path: str | Path = "configs/model.yaml",
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
    tracking_summary = _track_training_run(
        tracking_config=tracking_config,
        config=config,
        model_config_path=model_config_path,
        bundle_summary=bundle.summary.as_dict(),
        models=models,
        validation_predictions=validation_predictions,
        candidate=candidate,
        final_result=final_result,
        report=report,
    )
    summary = TrainingPipelineSummary(
        train_row_count=len(train_frame),
        validation_row_count=len(validation_frame),
        test_row_count=len(test_frame),
        feature_count=len(feature_columns),
        model_names=tuple(models),
        production_model=candidate.model_name,
        production_threshold=candidate.threshold,
        evaluation_report=str(config.evaluation_report),
        tracking_parent_run_id=tracking_summary.parent_run_id,
        tracking_candidate_run_id=tracking_summary.candidate_run_id,
        registered_model_version=tracking_summary.registered_model_version,
    )
    LOGGER.info("Model training completed: %s", summary)
    return summary


@dataclass(frozen=True)
class TrackingRunSummary:
    """References created by optional Sprint 3 MLflow tracking."""

    parent_run_id: str | None = None
    candidate_run_id: str | None = None
    registered_model_version: str | None = None


def _track_training_run(
    *,
    tracking_config: TrackingConfig | None,
    config: ModelConfig,
    model_config_path: str | Path,
    bundle_summary: dict[str, Any],
    models: dict[str, Any],
    validation_predictions: dict[str, tuple[Any, Any]],
    candidate: ProductionCandidate,
    final_result: FinalTestResult,
    report: dict[str, Any],
) -> TrackingRunSummary:
    """Log comparable model runs and register only the selected candidate."""

    if tracking_config is None:
        return TrackingRunSummary()

    client = MlflowTrackingClient(tracking_config)
    client.configure()
    metadata = build_run_metadata(
        dataset_summary=bundle_summary,
        target_column=config.target_column,
        config_path=model_config_path,
    )
    parent_tags = {
        "project": "transaction-risk-ml",
        "stage": "training",
        "dataset_name": metadata.dataset_name,
        "git_commit": metadata.git_commit,
    }
    training_parameters = {
        "random_seed": config.random_seed,
        "imbalance_strategy": config.imbalance_strategy,
        "preprocessing": "median_imputation+standard_scaling+one_hot_encoding",
        "threshold_candidates": list(config.thresholds),
    }
    with client.start_run(run_name="training-pipeline") as parent_run:
        client.set_tags(parent_tags)
        for model_name in models:
            target, probability = validation_predictions[model_name]
            evaluations = analyze_thresholds(target, probability, config.thresholds)
            selected = select_best_threshold(evaluations)
            with client.start_run(run_name=model_name, nested=True):
                client.set_tags(
                    {
                        "project": "transaction-risk-ml",
                        "stage": "validation",
                        "dataset_name": metadata.dataset_name,
                        "candidate_status": "candidate",
                        "git_commit": metadata.git_commit,
                    }
                )
                log_training_parameters(
                    client,
                    model_name=model_name,
                    config=tracking_config,
                    training_config={
                        **training_parameters,
                        "model": config.model_params[model_name],
                    },
                    dataset_summary={**metadata.as_dict(), **bundle_summary},
                )
                log_validation_metrics(
                    client,
                    model_name=model_name,
                    selected_evaluation=selected,
                    threshold_evaluations=evaluations,
                )

        with client.start_run(
            run_name=f"final-{candidate.model_name}", nested=True
        ) as candidate_run:
            client.set_tags(
                {
                    "project": "transaction-risk-ml",
                    "stage": "final",
                    "dataset_name": metadata.dataset_name,
                    "candidate_status": "candidate",
                    "git_commit": metadata.git_commit,
                }
            )
            candidate_evaluations = analyze_thresholds(
                validation_predictions[candidate.model_name][0],
                validation_predictions[candidate.model_name][1],
                config.thresholds,
            )
            candidate_validation = select_best_threshold(candidate_evaluations)
            log_training_parameters(
                client,
                model_name=candidate.model_name,
                config=tracking_config,
                training_config={
                    **training_parameters,
                    "model": config.model_params[candidate.model_name],
                },
                dataset_summary={**metadata.as_dict(), **bundle_summary},
            )
            log_validation_metrics(
                client,
                model_name=candidate.model_name,
                selected_evaluation=candidate_validation,
                threshold_evaluations=candidate_evaluations,
            )
            client.log_metrics(_final_test_metrics(final_result))
            client.log_dict(report, "evaluation/report.json")
            reference = log_and_register_model(
                client,
                models[candidate.model_name],
                model_name=candidate.model_name,
                artifact_path="model",
                registered_model_name=tracking_config.registered_model_name,
                version_tags={
                    "candidate_status": "candidate",
                    "model_name": candidate.model_name,
                    "validation.pr_auc": str(candidate.validation_metrics.pr_auc),
                    "test.pr_auc": str(final_result.metrics.pr_auc),
                    "test.f1": str(final_result.metrics.f1),
                    "dataset_name": metadata.dataset_name,
                    "git_commit": metadata.git_commit,
                    "config_hash": metadata.config_hash,
                },
            )
            client.log_dict(reference.as_dict(), "registry/model_reference.json")
            if reference.registered_model_version is None:
                raise RuntimeError("MLflow did not return a registered model version")
            client.set_model_alias(
                registered_model_name=tracking_config.registered_model_name,
                alias="candidate",
                version=reference.registered_model_version,
            )
            load_logged_model(client, reference)
            return TrackingRunSummary(
                parent_run_id=parent_run.info.run_id,
                candidate_run_id=candidate_run.info.run_id,
                registered_model_version=reference.registered_model_version,
            )


def _final_test_metrics(result: FinalTestResult) -> dict[str, float]:
    metrics = result.metrics
    return {
        "test.precision": metrics.precision,
        "test.recall": metrics.recall,
        "test.f1": metrics.f1,
        "test.roc_auc": metrics.roc_auc,
        "test.pr_auc": metrics.pr_auc,
        "test.selected_threshold": metrics.threshold,
    }


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
    parser.add_argument(
        "--tracking-config",
        default="configs/tracking.yaml",
        help="Path to MLflow tracking configuration",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = _parse_args()
    config = load_model_config(args.config)
    tracking_config = load_tracking_config(args.tracking_config)
    spark = SparkSession.builder.appName("TransactionRiskML-ModelTraining").getOrCreate()
    try:
        run_training_pipeline(
            spark,
            config,
            tracking_config=tracking_config,
            model_config_path=args.config,
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
