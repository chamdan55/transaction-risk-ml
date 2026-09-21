"""Explicitly promote an MLflow candidate model after a human quality-gate review."""

from __future__ import annotations

import argparse

from app.core.logging import setup_logging
from ml.tracking.client import MlflowTrackingClient
from ml.tracking.config import load_tracking_config
from ml.tracking.registry import LoggedModel, promote_registered_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Registered MLflow model version")
    parser.add_argument("--stage", required=True, choices=("staging", "production"))
    parser.add_argument("--approved-by", required=True, help="Reviewer approving the quality gate")
    parser.add_argument("--reason", required=True, help="Quality-gate decision rationale")
    parser.add_argument("--tracking-config", default="configs/tracking.yaml")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = _parse_args()
    tracking_config = load_tracking_config(args.tracking_config)
    client = MlflowTrackingClient(tracking_config)
    client.configure()
    promote_registered_model(
        client,
        LoggedModel(
            model_name="production-candidate",
            artifact_path="model",
            model_uri="",
            registered_model_name=tracking_config.registered_model_name,
            registered_model_version=args.version,
        ),
        stage=args.stage,
        approved_by=args.approved_by,
        reason=args.reason,
    )


if __name__ == "__main__":
    main()
