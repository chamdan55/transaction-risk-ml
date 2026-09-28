"""Static contract checks for load-test and kind assets; no cluster is required."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_load_scenario_exercises_valid_and_invalid_prediction_requests() -> None:
    scenario = (ROOT / "tests/load/transaction_risk.js").read_text(encoding="utf-8")

    assert "/v1/predictions" in scenario
    assert 'body.amount = "100.0"' in scenario
    assert "422" in scenario
    assert "p(95)<500" in scenario
    assert "p(99)<1000" in scenario


def test_kind_kustomization_includes_availability_primitives() -> None:
    kustomization = (ROOT / "deployment/kubernetes/kustomization.yaml").read_text(encoding="utf-8")
    deployment = (ROOT / "deployment/kubernetes/deployment.yaml").read_text(encoding="utf-8")

    for resource in (
        "namespace.yaml",
        "configmap.yaml",
        "deployment.yaml",
        "service.yaml",
        "pdb.yaml",
    ):
        assert resource in kustomization
    for setting in ("readinessProbe", "livenessProbe", "RollingUpdate", "emptyDir", "model-bundle"):
        assert setting in deployment


def test_retraining_cronjob_is_suspended_and_uses_guardrails() -> None:
    kustomization = (ROOT / "deployment/kubernetes/kustomization.yaml").read_text(encoding="utf-8")
    cronjob = (ROOT / "deployment/kubernetes/retraining-cronjob.yaml").read_text(encoding="utf-8")

    assert "retraining-cronjob.yaml" in kustomization
    assert "suspend: true" in cronjob
    assert "concurrencyPolicy: Forbid" in cronjob
    assert "backoffLimit: 0" in cronjob
    assert "approved-manifest" in cronjob
    assert "MLFLOW_TRACKING_URI" in cronjob
    assert "readOnlyRootFilesystem: true" in cronjob
    assert "mountPath: /workspace/spark-warehouse" in cronjob
