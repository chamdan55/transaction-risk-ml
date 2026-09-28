import json
import os
from pathlib import Path

import pytest

from ml.tracking.client import TrackingClientError
from pipelines import rollout_model
from pipelines.export_model_bundle import (
    _validate_report,
    _validate_version_tags,
    export_production_model_bundle,
)


def _make_bundle(root):
    bundle = root / "releases" / "random_forest-v7"
    models = bundle / "models"
    models.mkdir(parents=True)
    (models / "random_forest.joblib").write_bytes(b"model")
    (bundle / "evaluation_report.json").write_text(
        json.dumps(
            {
                "release": {
                    "registered_model_name": "transaction-risk-model",
                    "model_name": "random_forest",
                    "model_version": "7",
                }
            }
        ),
        encoding="utf-8",
    )
    return bundle


def test_model_bundle_metadata_is_tied_to_single_named_model(tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path)
    monkeypatch.setattr(rollout_model, "REPOSITORY_ROOT", tmp_path)

    relative_root, model_name, release = rollout_model._bundle_metadata(bundle)

    assert relative_root.as_posix() == "releases/random_forest-v7"
    assert model_name == "random_forest"
    assert release["model_version"] == "7"
    assert len(rollout_model._bundle_digest(bundle)) == 64


def test_model_export_helpers_require_approved_contract_and_report():
    tags = {
        "candidate_status": "production",
        "signature_validation": "passed",
        "serving_input_validation": "passed",
        "feature_contract_version": "pre-transaction-v1",
        "production_threshold": "0.5",
    }
    _validate_version_tags(tags)

    report = {
        "candidate": {"model_name": "random_forest"},
        "lineage": {"manifest_id": "manifest-1"},
        "config": {"feature_contract_version": "pre-transaction-v1"},
    }
    assert (
        _validate_report(
            report,
            tags=tags | {"model_name": "random_forest", "dataset_manifest_id": "manifest-1"},
        )
        == report
    )

    with pytest.raises(TrackingClientError, match="Only a production-approved"):
        _validate_version_tags(tags | {"candidate_status": "candidate"})


def test_model_export_disables_direct_minio_artifact_download(monkeypatch, tmp_path):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "true")
    monkeypatch.setenv("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "true")
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda _uri: None)
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda **_kwargs: object())

    with pytest.raises(TrackingClientError, match="Unable to resolve approved model alias"):
        export_production_model_bundle(
            tracking_config_path=Path(__file__).parents[2] / "configs" / "tracking-compose.yaml",
            tracking_uri=None,
            alias="production",
            output_root=tmp_path,
        )

    assert os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"] == "false"
    assert os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD"] == "false"


def test_successful_rollout_uses_versioned_bundle_and_readiness_gate(tmp_path, monkeypatch):
    bundle = _make_bundle(tmp_path)
    monkeypatch.setattr(rollout_model, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        rollout_model,
        "_validate_registry_release",
        lambda **_kwargs: (object(), "7"),
    )
    commands = []

    def record_command(command, **_kwargs):
        commands.append(command)
        if command[:2] == ["podman", "save"]:
            Path(command[command.index("--output") + 1]).write_bytes(b"image archive")

    monkeypatch.setattr(rollout_model, "_run", record_command)

    rollout_model.rollout_model_bundle(
        bundle_path=bundle,
        tracking_config_path="tracking.yaml",
        tracking_uri="http://mlflow:5000",
        namespace="transaction-risk",
        deployment="transaction-risk-api",
        cluster="transaction-risk",
    )

    assert commands[0][:2] == ["podman", "build"]
    assert "MODEL_BUNDLE_SOURCE=releases/random_forest-v7" in commands[0]
    image = commands[0][commands[0].index("--tag") + 1]
    assert image.startswith("localhost/transaction-risk-model-bundle:v7-")
    assert commands[1][:4] == ["podman", "save", "--format", "docker-archive"]
    assert commands[1][-1] == image
    archive = commands[1][commands[1].index("--output") + 1]
    assert commands[2] == ["kind", "load", "image-archive", archive, "--name", "transaction-risk"]
    assert not Path(archive).exists()
    patch = json.loads(commands[3][commands[3].index("--patch") + 1])
    assert patch["spec"]["template"]["spec"]["initContainers"][0]["image"] == image
    assert commands[4][3:5] == ["rollout", "status"]


@pytest.mark.parametrize(
    "failure_command,rollout_started",
    [("build", False), ("save", False), ("load", False), ("status", True)],
)
def test_failed_rollout_restores_registry_and_deployment_as_applicable(
    tmp_path,
    monkeypatch,
    failure_command,
    rollout_started,
):
    bundle = _make_bundle(tmp_path)
    monkeypatch.setattr(rollout_model, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        rollout_model,
        "_validate_registry_release",
        lambda **_kwargs: (object(), "7"),
    )

    def fail_selected_command(command, **_kwargs):
        if failure_command == "build" and command[:2] == ["podman", "build"]:
            raise rollout_model.ModelRolloutError("image build failed")
        if failure_command == "save" and command[:2] == ["podman", "save"]:
            raise rollout_model.ModelRolloutError("image export failed")
        if failure_command == "load" and command[:3] == ["kind", "load", "image-archive"]:
            raise rollout_model.ModelRolloutError("image import failed")
        if failure_command == "status" and "status" in command:
            raise rollout_model.ModelRolloutError("readiness timed out")

    restore_calls = []
    monkeypatch.setattr(rollout_model, "_run", fail_selected_command)
    monkeypatch.setattr(
        rollout_model,
        "_restore_previous_release",
        lambda **kwargs: restore_calls.append(kwargs) or [],
    )

    with pytest.raises(rollout_model.ModelRolloutError, match="rollout failed"):
        rollout_model.rollout_model_bundle(
            bundle_path=bundle,
            tracking_uri="http://mlflow:5000",
        )

    assert restore_calls[0]["model_version"] == "7"
    assert restore_calls[0]["rollout_started"] is rollout_started
