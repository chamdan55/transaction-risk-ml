"""Create a one-time approved manifest for a validated retraining dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.core.logging import setup_logging
from ml.data.spark import create_spark_session
from ml.tracking.lineage import build_dataset_manifest
from ml.training.config import load_model_config
from ml.training.dataset import load_training_dataset
from ml.training.retraining import (
    RetrainingError,
    create_approved_manifest,
    source_root,
    write_approved_manifest,
)

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--output", default="artifacts/retraining/approved-manifest.json")
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--reason", required=True)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = _parse_args()
    config_path = Path(args.config)
    config = load_model_config(config_path)
    if config.features_path.is_absolute():
        raise RetrainingError(
            "Approved retraining manifests require a repository-relative data.features_path"
        )
    spark = create_spark_session()
    try:
        bundle = load_training_dataset(
            spark,
            config.features_path,
            target_column=config.target_column,
        )
        manifest = build_dataset_manifest(
            dataset_summary=bundle.summary.as_dict(),
            dataset_path=config.features_path,
            config_path=config_path,
            target_column=config.target_column,
            feature_contract_version=config.feature_contract_version,
            repository_root=source_root(),
        )
        approved = create_approved_manifest(
            manifest.as_dict(),
            approved_by=args.approved_by,
            reason=args.reason,
        )
        write_approved_manifest(args.output, approved)
    finally:
        spark.stop()
    LOGGER.info(
        "Approved retraining manifest created: path=%s manifest_id=%s sha256=%s",
        args.output,
        manifest.manifest_id,
        approved["manifest_sha256"],
    )


if __name__ == "__main__":
    main()
