"""Exercise a remote MLflow run, proxied artifacts, and registry without moving aliases."""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
import uuid
from pathlib import Path

from ml.tracking.config import configure_artifact_transfers, load_tracking_config

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-config", default="configs/tracking-compose.yaml")
    parser.add_argument("--tracking-uri", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_tracking_config(args.tracking_config, tracking_uri=args.tracking_uri)
    configure_artifact_transfers(config)
    os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"

    try:
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise SystemExit(
            "MLflow client is missing; install the project's tracking dependencies"
        ) from exc

    client = MlflowClient(tracking_uri=config.uri)
    suffix = uuid.uuid4().hex[:12]
    experiment_name = f"trm016-smoke-{suffix}"
    model_name = f"{config.registered_model_name}-trm016-smoke-{suffix}"
    experiment_id = client.create_experiment(experiment_name)
    experiment = client.get_experiment(experiment_id)
    if experiment.artifact_location is None or not experiment.artifact_location.startswith(
        "mlflow-artifacts:/"
    ):
        raise RuntimeError(
            "The server did not assign a proxied MLflow artifact location to the new experiment"
        )
    run = client.create_run(experiment_id, run_name="artifact-round-trip")

    try:
        client.log_param(run.info.run_id, "smoke_ticket", "TRM-016")
        with (
            tempfile.TemporaryDirectory(prefix="trm016-source-") as source_dir,
            tempfile.TemporaryDirectory(prefix="trm016-download-") as download_dir,
        ):
            source = Path(source_dir) / "tracking-smoke.txt"
            source.write_text("MLflow proxied artifact round-trip passed.\n", encoding="utf-8")
            client.log_artifact(run.info.run_id, str(source))
            downloaded = Path(client.download_artifacts(run.info.run_id, source.name, download_dir))
            if downloaded.read_text(encoding="utf-8") != source.read_text(encoding="utf-8"):
                raise RuntimeError("Downloaded artifact content did not match the uploaded file")
        client.create_registered_model(model_name)
        registered_model = client.get_registered_model(model_name)
        client.set_terminated(run.info.run_id, status="FINISHED")
    except Exception:
        try:
            client.set_terminated(run.info.run_id, status="FAILED")
        except Exception:
            LOGGER.exception("Unable to mark MLflow tracking smoke run as failed")
        raise

    print(
        "TRM-016 tracking smoke passed: "
        f"experiment_id={experiment_id} run_id={run.info.run_id} "
        f"registered_model={registered_model.name}"
    )


if __name__ == "__main__":
    main()
