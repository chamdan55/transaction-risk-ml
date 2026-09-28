"""Reproducible Sprint 2 training pipeline with Sprint 3 MLflow tracking."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
from pyspark.sql import SparkSession

from app.core.logging import setup_logging
from ml.contracts.features import validate_model_feature_columns
from ml.data.spark import create_spark_session
from ml.evaluation.metrics import (
    EvaluationError,
    build_calibration_data,
    evaluate_binary_predictions,
)
from ml.evaluation.selection import (
    CandidateValidationResult,
    FinalTestResult,
    ProductionCandidate,
    evaluate_production_candidate_on_test,
    evaluate_validation_candidates,
    select_production_candidate_from_results,
)
from ml.evaluation.threshold import ThresholdEvaluation
from ml.tracking.client import MlflowTrackingClient, TrackingClientError
from ml.tracking.config import TrackingConfig, load_tracking_config
from ml.tracking.lineage import DatasetManifest, build_dataset_manifest
from ml.tracking.logging import log_training_parameters, log_validation_metrics
from ml.tracking.metadata import build_run_metadata
from ml.tracking.registry import load_logged_model, log_and_register_model
from ml.training.baseline import train_logistic_regression_baseline
from ml.training.config import ModelConfig, load_model_config
from ml.training.dataset import load_training_dataset
from ml.training.models import train_random_forest, train_xgboost
from ml.training.preprocessing import spark_frame_to_pandas
from ml.training.retraining import (
    RetrainingQualityPolicy,
    evaluate_retraining_quality,
    expected_calibration_error,
    validate_manifest_matches,
)
from ml.training.sampling import (
    SamplingSummary,
    deterministically_sample_rows,
    retain_all_positives_and_sample_negatives,
    summarize_rows,
)

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
    retraining_quality_gate_status: str | None = None
    retraining_failed_checks: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_training_pipeline(
    spark: SparkSession,
    config: ModelConfig,
    *,
    tracking_config: TrackingConfig | None = None,
    model_config_path: str | Path = "configs/model.yaml",
    approved_manifest: dict[str, Any] | None = None,
    retraining_policy: RetrainingQualityPolicy | None = None,
) -> TrainingPipelineSummary:
    """Train, select, and evaluate models from the configured feature splits."""

    LOGGER.info(
        "Starting model training pipeline: features_path=%s, target=%s, seed=%s",
        config.features_path,
        config.target_column,
        config.random_seed,
    )
    if (approved_manifest is None) != (retraining_policy is None):
        raise ValueError("approved_manifest and retraining_policy must be supplied together")
    if retraining_policy is not None and tracking_config is None:
        raise ValueError("Controlled retraining requires a remote MLflow tracking configuration")
    if retraining_policy is not None and not tracking_config.uri.startswith(
        ("http://", "https://")
    ):
        raise ValueError("Controlled retraining requires an HTTP(S) MLflow tracking server")

    bundle = load_training_dataset(spark, config.features_path)
    lineage_manifest = build_dataset_manifest(
        dataset_summary=bundle.summary.as_dict(),
        dataset_path=config.features_path,
        config_path=model_config_path,
        target_column=config.target_column,
        feature_contract_version=config.feature_contract_version,
        repository_root=Path(__file__).resolve().parents[1],
    )
    if approved_manifest is not None:
        validate_manifest_matches(approved_manifest, lineage_manifest.as_dict())
    feature_columns = bundle.summary.feature_columns
    LOGGER.info(
        "Training dataset contract loaded: train=%s, validation=%s, test=%s, features=%s",
        bundle.summary.split_summaries["train"].row_count,
        bundle.summary.split_summaries["validation"].row_count,
        bundle.summary.split_summaries["test"].row_count,
        len(feature_columns),
    )
    tracking_client = None
    incumbent_version = None
    incumbent_model = None
    if retraining_policy is not None:
        tracking_client = MlflowTrackingClient(tracking_config)
        tracking_client.configure()
        try:
            incumbent_version = tracking_client.get_model_version_by_alias(alias="production")
            if (
                incumbent_version.tags.get("feature_contract_version")
                != config.feature_contract_version
            ):
                raise ValueError(
                    "Production model feature contract does not match retraining config"
                )
            if incumbent_version.tags.get("signature_validation") != "passed":
                raise ValueError("Production model has no passing MLflow signature validation")
            if incumbent_version.tags.get("serving_input_validation") != "passed":
                raise ValueError("Production model has no passing serving-input validation")
            incumbent_model = tracking_client.load_model(
                f"models:/{tracking_config.registered_model_name}@production"
            )
            validate_model_feature_columns(tuple(incumbent_model.preprocessor.feature_columns))
        except (TrackingClientError, AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Controlled retraining requires a loadable, validated production alias"
            ) from exc

    sampled_splits, sampling_summaries = _sample_splits_for_training(bundle, config)
    train_frame = spark_frame_to_pandas(
        sampled_splits["train"], (*feature_columns, config.target_column)
    )
    validation_frame = spark_frame_to_pandas(
        sampled_splits["validation"], (*feature_columns, config.target_column)
    )
    test_frame = spark_frame_to_pandas(
        sampled_splits["test"], (*feature_columns, config.target_column)
    )

    LOGGER.info(
        "Sampled feature frames materialized: train=%s, validation=%s, test=%s",
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
    validation_results = evaluate_validation_candidates(
        validation_predictions,
        thresholds=config.thresholds,
        primary_metric=config.primary_metric,
        threshold_metric=config.threshold_metric,
        threshold_selection_strategy=config.threshold_selection_strategy,
        minimum_recall=config.minimum_recall,
        false_positive_cost=config.false_positive_cost,
        false_negative_cost=config.false_negative_cost,
    )
    _log_validation_threshold_evaluations(validation_results, config)
    try:
        candidate = select_production_candidate_from_results(
            validation_results,
            primary_metric=config.primary_metric,
            threshold_selection_strategy=config.threshold_selection_strategy,
            minimum_recall=config.minimum_recall,
        )
    except EvaluationError:
        if retraining_policy is None:
            raise
        candidate = _select_diagnostic_candidate(validation_results, config)
    LOGGER.info(
        "Production candidate selected: model=%s, threshold=%s, metric=%s, score=%s",
        candidate.model_name,
        candidate.threshold,
        candidate.selection_metric,
        candidate.validation_score,
    )
    validation_report = _build_validation_report(
        validation_results,
        validation_predictions,
        config,
    )

    test_target = test_frame[config.target_column]
    selected_model = models[candidate.model_name]
    test_probability = selected_model.predict_proba(test_frame)[:, 1]
    final_result = evaluate_production_candidate_on_test(
        candidate,
        test_target,
        test_probability,
        false_positive_cost=config.false_positive_cost,
        false_negative_cost=config.false_negative_cost,
    )
    LOGGER.info(
        "Final test evaluation completed: model=%s, threshold=%s, pr_auc=%.6f, f1=%.6f",
        candidate.model_name,
        candidate.threshold,
        final_result.metrics.pr_auc,
        final_result.metrics.f1,
    )
    retraining_gate = None
    if retraining_policy is not None:
        if incumbent_model is None or incumbent_version is None:
            raise RuntimeError("Production model was not loaded for retraining comparison")
        incumbent_probability = incumbent_model.predict_proba(test_frame)[:, 1]
        incumbent_threshold = _required_model_threshold(incumbent_version.tags)
        incumbent_test = evaluate_binary_predictions(
            test_target,
            incumbent_probability,
            threshold=incumbent_threshold,
        )
        incumbent_expected_cost = (
            incumbent_test.false_positives * config.false_positive_cost
            + incumbent_test.false_negatives * config.false_negative_cost
        )
        incumbent_calibration_error = expected_calibration_error(
            build_calibration_data(
                test_target,
                incumbent_probability,
                n_bins=config.calibration_bins,
            )
        )
        incumbent_metrics = {
            **incumbent_test.as_dict(),
            "expected_cost_per_transaction": float(incumbent_expected_cost / len(test_target)),
            "calibration_error": incumbent_calibration_error,
        }
        candidate_validation = _find_validation_result(validation_results, candidate.model_name)
        validation_evaluation = candidate_validation.selected_threshold or _diagnostic_threshold(
            candidate_validation, config
        )
        validation_metrics = {
            **validation_evaluation.metrics.as_dict(),
            "expected_cost": validation_evaluation.expected_cost,
        }
        candidate_calibration_error = expected_calibration_error(
            build_calibration_data(
                test_target,
                test_probability,
                n_bins=config.calibration_bins,
            )
        )
        retraining_gate = evaluate_retraining_quality(
            schema_contract_passed=(
                tuple(bundle.summary.feature_columns)
                == validate_model_feature_columns(bundle.summary.feature_columns)
            ),
            duplicate_transaction_id_count=sum(
                bundle.summary.duplicate_transaction_id_counts.values()
            ),
            split_overlap_count=sum(bundle.summary.overlapping_transaction_id_counts.values()),
            validation_metrics=validation_metrics,
            test_metrics={
                **final_result.metrics.as_dict(),
                "expected_cost_per_transaction": (final_result.expected_cost / len(test_target)),
            },
            test_calibration_error=candidate_calibration_error,
            test_row_count=len(test_frame),
            test_positive_count=int(test_target.sum()),
            incumbent_metrics=incumbent_metrics,
            policy=retraining_policy,
        )
        retraining_gate["incumbent_version"] = str(incumbent_version.version)
        retraining_gate["production_alias"] = "production"
        retraining_gate["approved_manifest_id"] = approved_manifest["manifest"]["manifest_id"]
        retraining_gate["approved_manifest_sha256"] = approved_manifest["manifest_sha256"]
        retraining_gate["approved_by"] = approved_manifest["approval"]["approved_by"]
        retraining_gate["policy_sha256"] = retraining_policy.policy_sha256
        report_retraining = retraining_gate
        LOGGER.info(
            "Retraining quality gate completed: status=%s failed_checks=%s incumbent_version=%s",
            retraining_gate["status"],
            retraining_gate["failed_checks"],
            retraining_gate["incumbent_version"],
        )
    else:
        report_retraining = None
    report = _build_report(
        config=config,
        bundle_summary=bundle.summary.as_dict(),
        sampling_summaries=sampling_summaries,
        validation_report=validation_report,
        candidate=candidate,
        final_result=final_result,
        lineage_manifest=lineage_manifest,
        test_calibration=build_calibration_data(
            test_target,
            test_probability,
            n_bins=config.calibration_bins,
        ),
    )
    if report_retraining is not None:
        report["retraining"] = report_retraining
    if approved_manifest is None:
        artifact_directory = config.model_directory
        evaluation_report_path = config.evaluation_report
        model_card_path = config.model_card
    else:
        run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_root = (
            Path("artifacts/retraining/runs")
            / f"{approved_manifest['manifest']['manifest_id'][:12]}-{run_stamp}-{uuid4().hex[:8]}"
        )
        artifact_directory = run_root / "models"
        evaluation_report_path = run_root / "evaluation_report.json"
        model_card_path = run_root / "model_card.md"
    _save_artifacts(models, config, output_directory=artifact_directory)
    LOGGER.info("Model artifacts saved: path=%s", artifact_directory)
    _write_json(evaluation_report_path, report)
    _write_model_card(model_card_path, config=config, report=report)
    LOGGER.info("Evaluation report saved: path=%s", evaluation_report_path)
    tracking_summary = _track_training_run(
        tracking_config=tracking_config,
        config=config,
        model_config_path=model_config_path,
        bundle_summary=bundle.summary.as_dict(),
        models=models,
        validation_predictions=validation_predictions,
        validation_results=validation_results,
        candidate=candidate,
        final_result=final_result,
        report=report,
        sampling_summaries=sampling_summaries,
        lineage_manifest=lineage_manifest,
        input_example=_build_serving_input_example(train_frame, feature_columns),
        retraining_quality_gate=retraining_gate,
        approved_manifest=approved_manifest,
        generated_model_card_path=model_card_path,
    )
    summary = TrainingPipelineSummary(
        train_row_count=len(train_frame),
        validation_row_count=len(validation_frame),
        test_row_count=len(test_frame),
        feature_count=len(feature_columns),
        model_names=tuple(models),
        production_model=candidate.model_name,
        production_threshold=candidate.threshold,
        evaluation_report=str(evaluation_report_path),
        tracking_parent_run_id=tracking_summary.parent_run_id,
        tracking_candidate_run_id=tracking_summary.candidate_run_id,
        registered_model_version=tracking_summary.registered_model_version,
        retraining_quality_gate_status=(
            retraining_gate["status"] if retraining_gate is not None else None
        ),
        retraining_failed_checks=(
            tuple(retraining_gate["failed_checks"]) if retraining_gate is not None else ()
        ),
    )
    LOGGER.info("Model training completed: %s", summary)
    return summary


def _sample_splits_for_training(
    bundle: Any,
    config: ModelConfig,
) -> tuple[dict[str, Any], dict[str, SamplingSummary]]:
    sampled_splits = {}
    summaries = {}
    train_df = bundle.by_name("train")
    if config.imbalance_strategy == "negative_sampling":
        sampled_df, summary = retain_all_positives_and_sample_negatives(
            train_df,
            target_column=config.target_column,
            max_rows=config.sampling_max_rows["train"],
            random_seed=config.random_seed,
        )
    else:
        sampled_df = train_df
        summary = summarize_rows(
            train_df,
            target_column=config.target_column,
            max_rows=config.sampling_max_rows["train"],
        )
    sampled_splits["train"] = sampled_df
    summaries["train"] = summary
    LOGGER.info("Sampling train split: %s", summary)

    for split_name in ("validation", "test"):
        split_df = bundle.by_name(split_name)
        sampling_strategy = (
            config.evaluation_sampling_strategy
            if split_name == "validation"
            else config.final_test_sampling_strategy
        )
        if sampling_strategy == "full":
            sampled_df = split_df
            summary = summarize_rows(
                split_df,
                target_column=config.target_column,
                max_rows=config.sampling_max_rows[split_name],
            )
        else:
            sampled_df, summary = deterministically_sample_rows(
                split_df,
                target_column=config.target_column,
                max_rows=config.sampling_max_rows[split_name],
                random_seed=config.random_seed,
                strategy=sampling_strategy,
            )
        sampled_splits[split_name] = sampled_df
        summaries[split_name] = summary
        LOGGER.info("Sampling %s split: %s", split_name, summary)
    return sampled_splits, summaries


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
    validation_results: tuple[CandidateValidationResult, ...],
    candidate: ProductionCandidate,
    final_result: FinalTestResult,
    report: dict[str, Any],
    sampling_summaries: dict[str, SamplingSummary],
    lineage_manifest: DatasetManifest,
    input_example: Any,
    retraining_quality_gate: dict[str, Any] | None,
    approved_manifest: dict[str, Any] | None,
    generated_model_card_path: Path,
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
        feature_contract_version=config.feature_contract_version,
        repository_root=Path.cwd(),
    )
    parent_tags = {
        "project": "transaction-risk-ml",
        "stage": "training",
        "dataset_name": metadata.dataset_name,
        "git_commit": metadata.git_commit,
        "dataset_manifest_id": lineage_manifest.manifest_id,
        "feature_contract_version": config.feature_contract_version,
    }
    if retraining_quality_gate is not None:
        parent_tags.update(
            {
                "stage": "retraining",
                "retraining.status": retraining_quality_gate["status"],
                "retraining.manifest_sha256": retraining_quality_gate["approved_manifest_sha256"],
                "retraining.incumbent_version": retraining_quality_gate["incumbent_version"],
                "retraining.policy_sha256": retraining_quality_gate["policy_sha256"],
            }
        )
    training_parameters = {
        "random_seed": config.random_seed,
        "imbalance_strategy": config.imbalance_strategy,
        "evaluation_sampling_strategy": config.evaluation_sampling_strategy,
        "final_test_sampling_strategy": config.final_test_sampling_strategy,
        "false_positive_cost": config.false_positive_cost,
        "false_negative_cost": config.false_negative_cost,
        "business_costs_are_assumptions": config.business_costs_are_assumptions,
        "minimum_recall": config.minimum_recall,
        "threshold_selection_strategy": config.threshold_selection_strategy,
        "threshold_metric": config.threshold_metric,
        "calibration_bins": config.calibration_bins,
        "feature_contract_version": config.feature_contract_version,
        "preprocessing": "median_imputation+standard_scaling+one_hot_encoding",
        "threshold_candidates": list(config.thresholds),
        "sampling": {
            split_name: summary.as_dict() for split_name, summary in sampling_summaries.items()
        },
    }
    with client.start_run(run_name="training-pipeline") as parent_run:
        client.set_tags(parent_tags)
        client.log_dict(lineage_manifest.as_dict(), "lineage/dataset_manifest.json")
        if approved_manifest is not None:
            client.log_dict(approved_manifest, "lineage/approved_dataset_manifest.json")
        for result in validation_results:
            model_name = result.model_name
            target, probability = validation_predictions[model_name]
            with client.start_run(run_name=model_name, nested=True):
                client.set_tags(
                    {
                        "project": "transaction-risk-ml",
                        "stage": "validation",
                        "dataset_name": metadata.dataset_name,
                        "candidate_status": "candidate" if result.is_eligible else "rejected",
                        "rejection_reason": result.rejection_reason or "",
                        "git_commit": metadata.git_commit,
                        "dataset_manifest_id": lineage_manifest.manifest_id,
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
                    selected_evaluation=result.selected_threshold,
                    threshold_evaluations=result.threshold_evaluations,
                    rejection_reason=result.rejection_reason,
                )
                client.log_dict(
                    {
                        "model_name": model_name,
                        "calibration": build_calibration_data(
                            target,
                            probability,
                            n_bins=config.calibration_bins,
                        ),
                    },
                    f"calibration/{model_name}.json",
                )

        with client.start_run(
            run_name=f"final-{candidate.model_name}", nested=True
        ) as candidate_run:
            client.set_tags(
                {
                    "project": "transaction-risk-ml",
                    "stage": "final",
                    "dataset_name": metadata.dataset_name,
                    "candidate_status": (
                        retraining_quality_gate["status"]
                        if retraining_quality_gate is not None
                        else "validation_pending"
                    ),
                    "git_commit": metadata.git_commit,
                }
            )
            candidate_validation = _find_validation_result(validation_results, candidate.model_name)
            if (
                candidate_validation.selected_threshold is None and retraining_quality_gate is None
            ):  # pragma: no cover - invariant
                raise RuntimeError("Selected production candidate is marked rejected")
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
                selected_evaluation=candidate_validation.selected_threshold,
                threshold_evaluations=candidate_validation.threshold_evaluations,
            )
            client.log_metrics(_final_test_metrics(final_result))
            if (
                retraining_quality_gate is not None
                and candidate_validation.selected_threshold is None
            ):
                client.log_metrics({"retraining.diagnostic_threshold": candidate.threshold})
            client.log_dict(report, "evaluation/report.json")
            if retraining_quality_gate is not None:
                gate_metrics = {
                    "retraining.quality_gate_passed": float(retraining_quality_gate["passed"]),
                    "incumbent.test.pr_auc": retraining_quality_gate["incumbent_metrics"]["pr_auc"],
                    "incumbent.test.recall": retraining_quality_gate["incumbent_metrics"]["recall"],
                    "incumbent.test.expected_cost_per_transaction": retraining_quality_gate[
                        "incumbent_metrics"
                    ]["expected_cost_per_transaction"],
                }
                client.log_metrics(gate_metrics)
                client.log_dict(retraining_quality_gate, "retraining/quality_gate_report.json")
                if not retraining_quality_gate["passed"]:
                    client.set_tags(
                        {
                            "candidate_status": "rejected",
                            "rejection_reason": ",".join(retraining_quality_gate["failed_checks"]),
                        }
                    )
            client.log_dict(
                {"markdown": generated_model_card_path.read_text(encoding="utf-8")},
                "governance/model_card.json",
            )
            client.log_dict(
                {
                    "feature_columns": list(lineage_manifest.feature_columns),
                    "feature_contract_version": config.feature_contract_version,
                    "input_example": input_example.to_dict(orient="records"),
                },
                "model/model_contract.json",
            )
            try:
                signature = client.infer_signature(models[candidate.model_name], input_example)
            except TrackingClientError:
                client.set_tags(
                    {
                        "candidate_status": "rejected",
                        "rejection_reason": "mlflow_signature_validation_failed",
                    }
                )
                raise
            client.log_dict(
                {
                    "status": "validated",
                    "feature_contract_version": config.feature_contract_version,
                    "input_example_columns": list(input_example.columns),
                    "signature": signature.to_dict(),
                },
                "model/signature_validation.json",
            )
            quality_rejected = (
                retraining_quality_gate is not None and not retraining_quality_gate["passed"]
            )
            if quality_rejected:
                model_info = client.log_model(
                    models[candidate.model_name],
                    artifact_path="model",
                    registered_model_name=None,
                    signature=signature,
                    input_example=input_example,
                )
                client.log_dict(
                    {
                        "model_name": candidate.model_name,
                        "artifact_path": "model",
                        "model_uri": model_info.model_uri,
                        "registered_model_name": tracking_config.registered_model_name,
                        "registered_model_version": None,
                    },
                    "registry/model_reference.json",
                )
                return TrackingRunSummary(
                    parent_run_id=parent_run.info.run_id,
                    candidate_run_id=candidate_run.info.run_id,
                    registered_model_version=None,
                )

            quality_tags = {}
            if retraining_quality_gate is not None:
                quality_tags = {
                    "retraining.quality_gate": "passed",
                    "retraining.incumbent_version": retraining_quality_gate["incumbent_version"],
                    "retraining.manifest_sha256": retraining_quality_gate[
                        "approved_manifest_sha256"
                    ],
                    "retraining.policy_sha256": retraining_quality_gate["policy_sha256"],
                }
            reference = log_and_register_model(
                client,
                models[candidate.model_name],
                model_name=candidate.model_name,
                artifact_path="model",
                registered_model_name=tracking_config.registered_model_name,
                signature=signature,
                input_example=input_example,
                version_tags={
                    "candidate_status": "candidate",
                    "model_name": candidate.model_name,
                    "validation.pr_auc": str(candidate.validation_metrics.pr_auc),
                    "test.pr_auc": str(final_result.metrics.pr_auc),
                    "test.f1": str(final_result.metrics.f1),
                    "dataset_name": metadata.dataset_name,
                    "git_commit": metadata.git_commit,
                    "config_hash": metadata.config_hash,
                    "dataset_manifest_id": lineage_manifest.manifest_id,
                    "schema_fingerprint": lineage_manifest.schema_fingerprint,
                    "content_fingerprint": lineage_manifest.content_fingerprint,
                    "config_fingerprint": lineage_manifest.config_fingerprint,
                    "code_fingerprint": lineage_manifest.code_fingerprint,
                    "feature_contract_version": config.feature_contract_version,
                    "production_threshold": str(candidate.threshold),
                    "threshold_selection_strategy": config.threshold_selection_strategy,
                    "evaluation_report_artifact": "evaluation/report.json",
                    "model_card_artifact": "governance/model_card.json",
                    "signature_validation": "passed",
                    "serving_input_validation": "pending",
                    **quality_tags,
                },
            )
            client.log_dict(reference.as_dict(), "registry/model_reference.json")
            if reference.registered_model_version is None:
                raise RuntimeError("MLflow did not return a registered model version")
            try:
                _validate_logged_model_serving_input(client, reference, input_example)
            except Exception as exc:
                client.set_tags(
                    {
                        "candidate_status": "rejected",
                        "rejection_reason": "logged_model_serving_validation_failed",
                    }
                )
                client.set_model_version_tags(
                    registered_model_name=tracking_config.registered_model_name,
                    version=reference.registered_model_version,
                    tags={
                        "candidate_status": "rejected",
                        "serving_input_validation": "failed",
                    },
                )
                raise RuntimeError(
                    "Logged candidate model failed feature-only serving input validation"
                ) from exc
            client.set_model_version_tags(
                registered_model_name=tracking_config.registered_model_name,
                version=reference.registered_model_version,
                tags={"serving_input_validation": "passed"},
            )
            client.set_tags({"candidate_status": "candidate"})
            client.set_model_alias(
                registered_model_name=tracking_config.registered_model_name,
                alias="candidate",
                version=reference.registered_model_version,
            )
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
        "test.brier_score": metrics.brier_score,
        "test.alert_rate": metrics.alert_rate,
        "test.expected_cost": result.expected_cost,
        "test.selected_threshold": metrics.threshold,
    }


def _find_validation_result(
    validation_results: tuple[CandidateValidationResult, ...], model_name: str
) -> CandidateValidationResult:
    for result in validation_results:
        if result.model_name == model_name:
            return result
    raise RuntimeError(f"Missing validation result for selected model: {model_name}")


def _diagnostic_threshold(
    result: CandidateValidationResult,
    config: ModelConfig,
) -> ThresholdEvaluation:
    """Choose the highest-recall threshold solely to record a rejected retraining attempt."""

    return max(
        result.threshold_evaluations,
        key=lambda item: (
            item.metrics.recall,
            getattr(item.metrics, config.primary_metric),
            -item.expected_cost,
            -item.metrics.threshold,
        ),
    )


def _select_diagnostic_candidate(
    results: tuple[CandidateValidationResult, ...],
    config: ModelConfig,
) -> ProductionCandidate:
    """Select a reportable model when all validation candidates fail a gate."""

    if not results:
        raise EvaluationError("No validation results are available for rejected retraining")
    selected_result = min(
        results,
        key=lambda item: (
            -getattr(_diagnostic_threshold(item, config).metrics, config.primary_metric),
            _diagnostic_threshold(item, config).expected_cost,
            item.model_name,
        ),
    )
    evaluation = _diagnostic_threshold(selected_result, config)
    return ProductionCandidate(
        model_name=selected_result.model_name,
        threshold=evaluation.metrics.threshold,
        selection_metric=config.primary_metric,
        validation_metrics=evaluation.metrics,
        validation_score=float(getattr(evaluation.metrics, config.primary_metric)),
        threshold_selection_strategy=config.threshold_selection_strategy,
        expected_cost=evaluation.expected_cost,
    )


def _required_model_threshold(tags: dict[str, str]) -> float:
    """Read and validate the serving threshold recorded on a production version."""

    try:
        threshold = float(tags["production_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Production model has no valid production_threshold tag") from exc
    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Production model production_threshold must be within [0, 1]")
    return threshold


def _build_serving_input_example(train_frame: Any, feature_columns: tuple[str, ...]) -> Any:
    """Build a feature-only example with nullable numeric columns represented as floats."""

    input_example = train_frame.loc[:, list(feature_columns)].head(2).copy()
    for column in input_example.select_dtypes(include="number").columns:
        input_example[column] = input_example[column].astype("float64")
    return input_example


def _validate_logged_model_serving_input(
    client: MlflowTrackingClient,
    reference: Any,
    input_example: Any,
) -> None:
    """Ensure the registered bundle accepts the exact feature-only MLflow example."""

    loaded_model = load_logged_model(client, reference)
    probabilities = loaded_model.predict_proba(input_example)
    predictions = loaded_model.predict(input_example)
    if len(probabilities) != len(input_example) or len(predictions) != len(input_example):
        raise RuntimeError("Loaded model returned a prediction count different from input rows")


def _build_validation_report(
    validation_results: tuple[CandidateValidationResult, ...],
    validation_predictions: dict[str, tuple[Any, Any]],
    config: ModelConfig,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for result in validation_results:
        target, probability = validation_predictions[result.model_name]
        report[result.model_name] = {
            **result.as_dict(),
            "calibration": build_calibration_data(
                target,
                probability,
                n_bins=config.calibration_bins,
            ),
        }
    return report


def _log_validation_threshold_evaluations(
    validation_results: tuple[CandidateValidationResult, ...], config: ModelConfig
) -> None:
    """Log validation metrics before the production quality gate is applied.

    The pipeline deliberately stops when no threshold meets ``minimum_recall``.
    Emitting the complete threshold analysis first keeps that failure actionable
    in a local console or centralized log stream.
    """

    LOGGER.info(
        "Validation threshold evaluation started: strategy=%s, minimum_recall=%s, thresholds=%s",
        config.threshold_selection_strategy,
        config.minimum_recall,
        config.thresholds,
    )
    for result in validation_results:
        ranking_metrics = result.threshold_evaluations[0].metrics
        LOGGER.info(
            "Validation model ranking: model=%s, roc_auc=%.6f, pr_auc=%.6f, brier_score=%.6f",
            result.model_name,
            ranking_metrics.roc_auc,
            ranking_metrics.pr_auc,
            ranking_metrics.brier_score,
        )
        for evaluation in result.threshold_evaluations:
            metrics = evaluation.metrics
            passes_recall_gate = (
                config.minimum_recall is None or metrics.recall >= config.minimum_recall
            )
            LOGGER.info(
                "Validation threshold: model=%s, threshold=%.4f, precision=%.6f, "
                "recall=%.6f, f1=%.6f, alert_rate=%.6f, expected_cost=%.2f, "
                "recall_gate=%s",
                result.model_name,
                metrics.threshold,
                metrics.precision,
                metrics.recall,
                metrics.f1,
                metrics.alert_rate,
                evaluation.expected_cost,
                "PASS" if passes_recall_gate else "FAIL",
            )
        if not result.is_eligible:
            LOGGER.warning(
                "Validation quality gate unmet: model=%s, minimum_recall=%.6f, "
                "maximum_candidate_recall=%.6f",
                result.model_name,
                config.minimum_recall,
                result.maximum_candidate_recall,
            )


def _build_report(
    *,
    config: ModelConfig,
    bundle_summary: dict[str, Any],
    sampling_summaries: dict[str, SamplingSummary],
    validation_report: dict[str, Any],
    candidate: ProductionCandidate,
    final_result: FinalTestResult,
    lineage_manifest: DatasetManifest,
    test_calibration: list[dict[str, float | int]],
) -> dict[str, Any]:
    return {
        "config": {
            "target_column": config.target_column,
            "feature_contract_version": config.feature_contract_version,
            "random_seed": config.random_seed,
            "primary_metric": config.primary_metric,
            "imbalance_strategy": config.imbalance_strategy,
            "evaluation_sampling_strategy": config.evaluation_sampling_strategy,
            "final_test_sampling_strategy": config.final_test_sampling_strategy,
            "false_positive_cost": config.false_positive_cost,
            "false_negative_cost": config.false_negative_cost,
            "business_costs_are_assumptions": config.business_costs_are_assumptions,
            "minimum_recall": config.minimum_recall,
            "threshold_selection_strategy": config.threshold_selection_strategy,
            "threshold_metric": config.threshold_metric,
            "calibration_bins": config.calibration_bins,
            "thresholds": list(config.thresholds),
        },
        "dataset": bundle_summary,
        "lineage": lineage_manifest.stable_dict(),
        "sampling": {
            split_name: summary.as_dict() for split_name, summary in sampling_summaries.items()
        },
        "validation": validation_report,
        "candidate": candidate.as_dict(),
        "test": {
            **final_result.metrics.as_dict(),
            "expected_cost": final_result.expected_cost,
            "calibration": test_calibration,
            "sampling_policy": config.final_test_sampling_strategy,
        },
    }


def _save_artifacts(
    models: dict[str, Any],
    config: ModelConfig,
    *,
    output_directory: Path | None = None,
) -> None:
    model_directory = output_directory or config.model_directory
    model_directory.mkdir(parents=True, exist_ok=True)
    for model_name, model in models.items():
        try:
            validate_model_feature_columns(tuple(model.preprocessor.feature_columns))
        except (AttributeError, ValueError) as exc:
            raise ValueError(
                f"Model artifact {model_name} does not use the approved feature contract"
            ) from exc
        joblib.dump(model, model_directory / f"{model_name}.joblib")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _write_model_card(path: Path, *, config: ModelConfig, report: dict[str, Any]) -> None:
    """Write a concise, versionable governance summary without auto-promotion claims."""

    candidate = report["candidate"]
    test = report["test"]
    assumptions = "assumption" if config.business_costs_are_assumptions else "business-approved"
    content = f"""# Transaction Risk Model Card

## Intended use

- Candidate model: `{candidate["model_name"]}`
- Feature contract: `{config.feature_contract_version}`
- Primary selection metric: `{config.primary_metric}`
- Threshold policy: `{config.threshold_selection_strategy}` (`{config.threshold_metric}`)
- Final test sampling: `{config.final_test_sampling_strategy}`

## Validation and final test

- Selected threshold: `{candidate["threshold"]}`
- Validation score: `{candidate["validation_score"]}`
- Final test PR-AUC: `{test["pr_auc"]}`
- Final test ROC-AUC: `{test["roc_auc"]}`
- Final test precision/recall/F1: `{test["precision"]}` / `{test["recall"]}` / `{test["f1"]}`
- Final test Brier score: `{test["brier_score"]}`
- Final test alert rate: `{test["alert_rate"]}`
- Final test expected cost: `{test["expected_cost"]}`

## Business assumptions and limitations

- False-positive cost: `{config.false_positive_cost}` ({assumptions}).
- False-negative cost: `{config.false_negative_cost}` ({assumptions}).
- Minimum recall constraint: `{config.minimum_recall}`.
- Calibration must be reviewed before interpreting scores as probabilities; calibration bins are
  included in the evaluation report.
- The test set is evaluated once after validation-based model and threshold selection.
- This artifact is a candidate model card. It does not authorize automatic production promotion.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


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
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Override the tracking URI from the selected config for non-default host ports",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = _parse_args()
    config = load_model_config(args.config)
    tracking_config = load_tracking_config(
        args.tracking_config,
        tracking_uri=args.tracking_uri,
    )
    spark = create_spark_session()
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
