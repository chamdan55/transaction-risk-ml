"""Run an approved, production-compared retraining attempt."""

from __future__ import annotations

import argparse
import logging
from typing import Any

from app.core.logging import setup_logging
from ml.data.spark import create_spark_session
from ml.tracking.client import MlflowTrackingClient
from ml.tracking.config import load_tracking_config
from ml.training.config import load_model_config
from ml.training.retraining import (
    RetrainingError,
    load_approved_manifest,
    load_quality_policy,
)
from pipelines.train_models import run_training_pipeline

LOGGER = logging.getLogger(__name__)


class _RecordedFailedAttempt(Exception):
    """Internal sentinel used to make the diagnostic MLflow run end in FAILED state."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-manifest", required=True)
    parser.add_argument("--policy", default="configs/retraining.yaml")
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--tracking-config", default="configs/tracking-compose.yaml")
    parser.add_argument("--tracking-uri", default=None)
    return parser.parse_args()


def _record_failed_attempt(
    *,
    tracking_client: MlflowTrackingClient,
    approved_manifest: dict[str, Any],
    policy_sha256: str,
    error: Exception,
) -> None:
    """Record failed preflight/training attempts without allowing logging to mask the failure."""

    try:
        tracking_client.configure()
        try:
            with tracking_client.start_run(run_name="controlled-retraining-failed"):
                tracking_client.set_tags(
                    {
                        "project": "transaction-risk-ml",
                        "stage": "retraining",
                        "candidate_status": "failed",
                        "rejection_reason": type(error).__name__,
                        "dataset_manifest_id": approved_manifest["manifest"]["manifest_id"],
                        "retraining.manifest_sha256": approved_manifest["manifest_sha256"],
                        "retraining.approved_by": approved_manifest["approval"]["approved_by"],
                        "retraining.policy_sha256": policy_sha256,
                    }
                )
                tracking_client.log_dict(
                    approved_manifest,
                    "lineage/approved_dataset_manifest.json",
                )
                tracking_client.log_dict(
                    {
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "message": str(error),
                    },
                    "retraining/failure.json",
                )
                raise _RecordedFailedAttempt from error
        except _RecordedFailedAttempt:
            return
    except Exception:
        LOGGER.exception("Unable to record the failed retraining attempt in MLflow")


def main() -> None:
    setup_logging()
    args = _parse_args()
    approved_manifest = load_approved_manifest(args.approved_manifest)
    policy = load_quality_policy(args.policy)
    model_config = load_model_config(args.config)
    tracking_config = load_tracking_config(
        args.tracking_config,
        tracking_uri=args.tracking_uri,
    )
    if not tracking_config.uri.startswith(("http://", "https://")):
        raise RetrainingError("Controlled retraining requires a remote HTTP(S) tracking URI")

    spark = create_spark_session()
    tracking_client = MlflowTrackingClient(tracking_config)
    try:
        summary = run_training_pipeline(
            spark,
            model_config,
            tracking_config=tracking_config,
            model_config_path=args.config,
            approved_manifest=approved_manifest,
            retraining_policy=policy,
        )
        LOGGER.info(
            "Controlled retraining completed: status=%s failed_checks=%s candidate_version=%s "
            "manifest_id=%s",
            summary.retraining_quality_gate_status,
            summary.retraining_failed_checks,
            summary.registered_model_version,
            approved_manifest["manifest"]["manifest_id"],
        )
        if summary.retraining_quality_gate_status != "passed":
            raise SystemExit(2)
    except SystemExit:
        raise
    except Exception as exc:
        _record_failed_attempt(
            tracking_client=tracking_client,
            approved_manifest=approved_manifest,
            policy_sha256=policy.policy_sha256,
            error=exc,
        )
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
