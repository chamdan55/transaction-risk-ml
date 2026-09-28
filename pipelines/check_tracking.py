"""Read-only connectivity check for the configured MLflow tracking server."""

from __future__ import annotations

import argparse
import logging

from ml.tracking.config import load_tracking_config

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-config", default="configs/tracking-compose.yaml")
    parser.add_argument("--tracking-uri", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    config = load_tracking_config(args.tracking_config, tracking_uri=args.tracking_uri)

    try:
        from mlflow.tracking import MlflowClient

        experiments = MlflowClient(tracking_uri=config.uri).search_experiments(max_results=1)
    except ImportError as exc:
        raise SystemExit(
            "MLflow client is missing; install the project's tracking dependencies"
        ) from exc
    except Exception as exc:
        raise SystemExit(
            "MLflow tracking server is unavailable or rejected the request "
            f"({type(exc).__name__}). Start the tracking profile and verify MLFLOW_PORT."
        ) from exc

    LOGGER.info(
        "MLflow tracking server is reachable; visible experiment sample=%d", len(experiments)
    )


if __name__ == "__main__":
    main()
