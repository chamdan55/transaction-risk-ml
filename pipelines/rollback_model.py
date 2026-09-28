"""Restore the previous model version recorded for a registry alias."""

from __future__ import annotations

import argparse
import logging

from app.core.logging import setup_logging
from ml.tracking.client import MlflowTrackingClient
from ml.tracking.config import load_tracking_config
from ml.tracking.registry import rollback_registered_model

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("staging", "production"))
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--tracking-config", default="configs/tracking-compose.yaml")
    parser.add_argument("--tracking-uri", default=None)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = _parse_args()
    config = load_tracking_config(args.tracking_config, tracking_uri=args.tracking_uri)
    if not config.uri.startswith(("http://", "https://")):
        raise ValueError("Controlled model rollback requires a remote HTTP(S) tracking URI")
    client = MlflowTrackingClient(config)
    client.configure()
    record = rollback_registered_model(
        client,
        stage=args.stage,
        approved_by=args.approved_by,
        reason=args.reason,
    )
    LOGGER.warning(
        "Model alias rollback completed: model=%s alias=%s from_version=%s to_version=%s "
        "approved_by=%s timestamp_utc=%s reason=%s",
        record.registered_model_name,
        record.alias,
        record.source_version,
        record.restored_version,
        record.approved_by,
        record.timestamp_utc,
        record.reason,
    )


if __name__ == "__main__":
    main()
