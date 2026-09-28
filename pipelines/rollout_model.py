"""Roll a promoted model bundle into kind and restore the prior release on failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from app.core.logging import setup_logging
from ml.tracking.client import MlflowTrackingClient, TrackingClientError
from ml.tracking.config import load_tracking_config
from ml.tracking.registry import rollback_registered_model

LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_BUNDLE_IMAGE = "localhost/transaction-risk-model-bundle"


class ModelRolloutError(RuntimeError):
    """Raised when a model bundle cannot be rolled out or safely rolled back."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-path", required=True)
    parser.add_argument(
        "--image-tag", default=None, help="Optional prefix; bundle digest is appended"
    )
    parser.add_argument("--tracking-config", default="configs/tracking-compose.yaml")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--namespace", default="transaction-risk")
    parser.add_argument("--deployment", default="transaction-risk-api")
    parser.add_argument("--cluster", default="transaction-risk")
    parser.add_argument("--timeout", default="3m")
    return parser.parse_args()


def _run(command: Sequence[str], *, timeout_seconds: float | None = None) -> None:
    LOGGER.info("Running deployment command: %s", " ".join(command))
    try:
        subprocess.run(
            list(command),
            cwd=REPOSITORY_ROOT,
            check=True,
            timeout=timeout_seconds,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        diagnostic = (exc.stderr or exc.stdout or "").strip()[-1200:]
        raise ModelRolloutError(
            f"Deployment command failed with exit code {exc.returncode}: {diagnostic}"
        ) from exc


def _bundle_metadata(bundle_path: str | Path) -> tuple[Path, str, dict[str, object]]:
    root = Path(bundle_path).resolve()
    try:
        relative_root = root.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ModelRolloutError("Model bundle must be inside the repository build context") from exc
    report_path = root / "evaluation_report.json"
    model_files = tuple((root / "models").glob("*.joblib"))
    if not report_path.is_file() or len(model_files) != 1:
        raise ModelRolloutError(
            "Model bundle must contain evaluation_report.json and exactly one models/*.joblib"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelRolloutError("Model bundle evaluation_report.json is invalid") from exc
    if not isinstance(report, dict) or not isinstance(report.get("release"), dict):
        raise ModelRolloutError("Model bundle has no production release metadata")
    release = report["release"]
    model_name = release.get("model_name")
    if (
        not isinstance(model_name, str)
        or not re.fullmatch(r"[a-zA-Z0-9_-]+", model_name)
        or model_files[0].stem != model_name
    ):
        raise ModelRolloutError("Model bundle artifact does not match its release model name")
    return relative_root, model_name, release


def _bundle_digest(bundle_path: str | Path) -> str:
    digest = hashlib.sha256()
    root = Path(bundle_path)
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as bundle_file:
            for chunk in iter(lambda: bundle_file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _validate_registry_release(
    *,
    tracking_config_path: str | Path,
    tracking_uri: str | None,
    release: dict[str, object],
) -> tuple[MlflowTrackingClient, str]:
    config = load_tracking_config(tracking_config_path, tracking_uri=tracking_uri)
    if not config.uri.startswith(("http://", "https://")):
        raise ModelRolloutError("Model rollout requires a remote HTTP(S) tracking server")
    client = MlflowTrackingClient(config)
    client.configure()
    version = str(release.get("model_version", ""))
    model_name = release.get("registered_model_name")
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", version)
        or model_name != config.registered_model_name
    ):
        raise ModelRolloutError("Model bundle registry identity does not match tracking config")
    try:
        active = client.get_model_version_by_alias(alias="production")
        if str(active.version) != version:
            raise ModelRolloutError(
                f"Production alias points to version {active.version}, but bundle is version {version}"
            )
        if active.tags.get("candidate_status") != "production":
            raise ModelRolloutError("Active production model is not marked as approved")
        previous_production_version = active.tags.get(
            "promotion.previous_version.production",
            active.tags.get("promotion.previous_version"),
        )
        if not previous_production_version or previous_production_version == "none":
            raise ModelRolloutError(
                "Production rollout requires a previously promoted version for automatic rollback"
            )
    except TrackingClientError:
        raise
    return client, version


def _restore_previous_release(
    *,
    client: MlflowTrackingClient,
    model_version: str,
    namespace: str,
    deployment: str,
    timeout: str,
    rollout_started: bool,
) -> list[str]:
    errors = []
    if rollout_started:
        try:
            _run(["kubectl", "-n", namespace, "rollout", "undo", f"deployment/{deployment}"])
            _run(
                [
                    "kubectl",
                    "-n",
                    namespace,
                    "rollout",
                    "status",
                    f"deployment/{deployment}",
                    f"--timeout={timeout}",
                ],
                timeout_seconds=240,
            )
        except Exception as exc:
            errors.append(f"Kubernetes rollback/readiness failed: {exc}")
    try:
        record = rollback_registered_model(
            client,
            stage="production",
            approved_by="automated-rollout-guard",
            reason=f"Kubernetes rollout for model version {model_version} failed readiness",
            expected_current_version=model_version,
        )
        LOGGER.warning(
            "Registry production alias restored after failed rollout: from=%s to=%s",
            record.source_version,
            record.restored_version,
        )
    except Exception as exc:
        errors.append(f"MLflow production alias rollback failed: {exc}")
    return errors


def rollout_model_bundle(
    *,
    bundle_path: str | Path,
    image_tag: str | None = None,
    tracking_config_path: str | Path = "configs/tracking-compose.yaml",
    tracking_uri: str | None = None,
    namespace: str = "transaction-risk",
    deployment: str = "transaction-risk-api",
    cluster: str = "transaction-risk",
    timeout: str = "3m",
) -> None:
    """Build/load a tagged bundle and roll it out with automatic dual rollback on failure."""

    if image_tag is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", image_tag):
        raise ModelRolloutError("image_tag must be a safe immutable image tag")
    relative_bundle, model_name, release = _bundle_metadata(bundle_path)
    client, version = _validate_registry_release(
        tracking_config_path=tracking_config_path,
        tracking_uri=tracking_uri,
        release=release,
    )
    prefix = image_tag or f"v{version}"
    image_tag = f"{prefix}-{_bundle_digest(bundle_path)[:12]}"
    image = f"{MODEL_BUNDLE_IMAGE}:{image_tag}"
    patch = {
        "spec": {
            "template": {
                "spec": {
                    "initContainers": [{"name": "model-bundle", "image": image}],
                    "containers": [
                        {
                            "name": "api",
                            "env": [
                                {
                                    "name": "MODEL_ARTIFACT_PATH",
                                    "value": f"/models/{model_name}.joblib",
                                }
                            ],
                        }
                    ],
                }
            }
        }
    }
    rollout_started = False
    try:
        _run(
            [
                "podman",
                "build",
                "--target",
                "model-bundle",
                "--build-arg",
                f"MODEL_BUNDLE_SOURCE={relative_bundle.as_posix()}",
                "--tag",
                image,
                ".",
            ]
        )
        # kind's docker-image loader checks the Docker CLI even with its Podman
        # provider. Export from Podman and import the archive into kind instead.
        with tempfile.TemporaryDirectory(prefix="trm-kind-model-") as temporary_dir:
            archive = str(Path(temporary_dir) / "model-bundle.tar")
            _run(["podman", "save", "--format", "docker-archive", "--output", archive, image])
            _run(["kind", "load", "image-archive", archive, "--name", cluster])
        _run(
            [
                "kubectl",
                "-n",
                namespace,
                "patch",
                f"deployment/{deployment}",
                "--type=strategic",
                "--patch",
                json.dumps(patch, separators=(",", ":")),
            ]
        )
        rollout_started = True
        _run(
            [
                "kubectl",
                "-n",
                namespace,
                "rollout",
                "status",
                f"deployment/{deployment}",
                f"--timeout={timeout}",
            ],
            timeout_seconds=240,
        )
    except Exception as exc:
        rollback_errors = _restore_previous_release(
            client=client,
            model_version=version,
            namespace=namespace,
            deployment=deployment,
            timeout=timeout,
            rollout_started=rollout_started,
        )
        if rollback_errors:
            detail = "; ".join(rollback_errors)
        elif rollout_started:
            detail = "Kubernetes revision and registry alias restored"
        else:
            detail = "registry alias restored; Kubernetes deployment was unchanged"
        raise ModelRolloutError(
            f"Model version {version} rollout failed: {exc}; rollback result: {detail}"
        ) from exc
    LOGGER.info(
        "Model rollout is Ready: version=%s model=%s image=%s namespace=%s deployment=%s",
        version,
        model_name,
        image,
        namespace,
        deployment,
    )


def main() -> None:
    setup_logging()
    args = _parse_args()
    rollout_model_bundle(
        bundle_path=args.bundle_path,
        image_tag=args.image_tag,
        tracking_config_path=args.tracking_config,
        tracking_uri=args.tracking_uri,
        namespace=args.namespace,
        deployment=args.deployment,
        cluster=args.cluster,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
