"""Export a production MLflow alias as a versioned, inference-only model bundle."""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import joblib

from app.core.logging import setup_logging
from ml.contracts.features import FEATURE_CONTRACT_VERSION, validate_model_feature_columns
from ml.tracking.client import TrackingClientError
from ml.tracking.config import configure_artifact_transfers, load_tracking_config

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-config", default="configs/tracking-compose.yaml")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--alias", default="production")
    parser.add_argument("--output-root", default="artifacts/model-bundles")
    return parser.parse_args()


def _validate_version_tags(tags: dict[str, str]) -> None:
    if tags.get("candidate_status") != "production":
        raise TrackingClientError("Only a production-approved model version can be exported")
    if tags.get("signature_validation") != "passed":
        raise TrackingClientError("Model version has not passed signature validation")
    if tags.get("serving_input_validation") != "passed":
        raise TrackingClientError("Model version has not passed serving-input validation")
    if tags.get("feature_contract_version") != FEATURE_CONTRACT_VERSION:
        raise TrackingClientError("Model version feature contract does not match this API")
    try:
        threshold = float(tags["production_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrackingClientError("Model version has no valid production threshold") from exc
    if not 0 <= threshold <= 1:
        raise TrackingClientError("Model version production threshold is outside [0, 1]")


def _validate_report(report: Any, *, tags: dict[str, str]) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise TrackingClientError("Model evaluation artifact must contain a JSON object")
    try:
        candidate = report["candidate"]
        lineage = report["lineage"]
        config = report["config"]
    except KeyError as exc:
        raise TrackingClientError(f"Model evaluation report is missing {exc.args[0]}") from exc
    if candidate.get("model_name") != tags.get("model_name"):
        raise TrackingClientError("Model evaluation report does not match registry model_name")
    if lineage.get("manifest_id") != tags.get("dataset_manifest_id"):
        raise TrackingClientError("Model evaluation report does not match registry lineage")
    if config.get("feature_contract_version") != FEATURE_CONTRACT_VERSION:
        raise TrackingClientError("Evaluation report feature contract does not match this API")
    return report


def export_production_model_bundle(
    *,
    tracking_config_path: str | Path,
    tracking_uri: str | None,
    alias: str,
    output_root: str | Path,
) -> Path:
    """Download and validate one promoted model into a new immutable release directory."""

    tracking_config = load_tracking_config(tracking_config_path, tracking_uri=tracking_uri)
    if not tracking_config.uri.startswith(("http://", "https://")):
        raise TrackingClientError("Model bundle export requires an HTTP(S) MLflow tracking server")
    configure_artifact_transfers(tracking_config)
    try:
        import mlflow
    except ImportError as exc:
        raise TrackingClientError(
            "MLflow tracking dependencies are required for model export"
        ) from exc

    mlflow.set_tracking_uri(tracking_config.uri)
    registry_client = mlflow.tracking.MlflowClient(tracking_uri=tracking_config.uri)
    try:
        model_version = registry_client.get_model_version_by_alias(
            tracking_config.registered_model_name,
            alias,
        )
    except Exception as exc:
        raise TrackingClientError(
            f"Unable to resolve approved model alias: {tracking_config.registered_model_name}@{alias}"
        ) from exc
    tags = dict(model_version.tags)
    _validate_version_tags(tags)
    if not model_version.run_id:
        raise TrackingClientError("Registered model version does not reference its source run")

    model_uri = f"models:/{tracking_config.registered_model_name}@{alias}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as exc:
        raise TrackingClientError(f"Unable to load approved MLflow model alias: {alias}") from exc
    try:
        validate_model_feature_columns(tuple(model.preprocessor.feature_columns))
    except (AttributeError, TypeError, ValueError) as exc:
        raise TrackingClientError(
            "Production model features do not match the serving contract"
        ) from exc

    try:
        with tempfile.TemporaryDirectory(prefix="trm013-report-") as temporary_dir:
            downloaded_report = Path(
                registry_client.download_artifacts(
                    model_version.run_id,
                    tags.get("evaluation_report_artifact", "evaluation/report.json"),
                    dst_path=temporary_dir,
                )
            )
            report = _validate_report(
                json.loads(downloaded_report.read_text(encoding="utf-8")),
                tags=tags,
            )
    except TrackingClientError:
        raise
    except Exception as exc:
        raise TrackingClientError(
            "Unable to download or validate the model evaluation report"
        ) from exc

    model_name = tags.get("model_name", "")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", model_name):
        raise TrackingClientError("Registry model_name is not a safe artifact filename")
    release_dir = Path(output_root) / f"{model_name}-v{model_version.version}"
    try:
        release_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise TrackingClientError(
            f"Refusing to overwrite existing model release bundle: {release_dir}"
        ) from exc

    try:
        models_dir = release_dir / "models"
        models_dir.mkdir()
        joblib.dump(model, models_dir / f"{model_name}.joblib")
        report["release"] = {
            "registered_model_name": tracking_config.registered_model_name,
            "model_name": model_name,
            "model_version": str(model_version.version),
            "alias": alias,
            "model_uri": model_uri,
            "source_run_id": model_version.run_id,
        }
        (release_dir / "evaluation_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(release_dir, ignore_errors=True)
        raise
    return release_dir


def main() -> None:
    setup_logging()
    args = _parse_args()
    release_dir = export_production_model_bundle(
        tracking_config_path=args.tracking_config,
        tracking_uri=args.tracking_uri,
        alias=args.alias,
        output_root=args.output_root,
    )
    LOGGER.info("Production model bundle exported: path=%s", release_dir)


if __name__ == "__main__":
    main()
