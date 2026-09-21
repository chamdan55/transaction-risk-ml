import importlib.util

import pytest
from sklearn.linear_model import LogisticRegression

from ml.tracking.client import MlflowTrackingClient
from ml.tracking.config import TrackingConfig
from ml.tracking.registry import (
    load_logged_model,
    log_and_register_model,
    promote_registered_model,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mlflow") is None,
    reason="mlflow is required for the MLflow integration test",
)


def test_registered_model_is_loadable_and_can_be_explicitly_promoted(tmp_path):
    import mlflow

    database_path = (tmp_path / "mlflow.db").as_posix()
    config = TrackingConfig(
        uri=f"sqlite:///{database_path}",
        experiment_name="transaction-risk-classification",
        registered_model_name="transaction-risk-model",
        artifact_location=(tmp_path / "artifacts").as_uri(),
    )
    client = MlflowTrackingClient(config)
    client.configure()
    model = LogisticRegression(random_state=42).fit([[0], [1]], [0, 1])

    with client.start_run(run_name="integration"):
        client.set_tags({"project": "transaction-risk-ml", "candidate_status": "candidate"})
        reference = log_and_register_model(
            client,
            model,
            model_name="logistic_regression",
            registered_model_name=config.registered_model_name,
            version_tags={"candidate_status": "candidate", "test.pr_auc": "1.0"},
        )
        loaded = load_logged_model(client, reference)

    assert reference.registered_model_version is not None
    assert loaded.predict([[0]]).shape == (1,)

    registry_client = mlflow.tracking.MlflowClient(tracking_uri=config.uri)
    version = registry_client.get_model_version(
        config.registered_model_name,
        reference.registered_model_version,
    )
    assert version.tags["candidate_status"] == "candidate"

    promote_registered_model(
        client,
        reference,
        stage="staging",
        approved_by="integration-test",
        reason="Explicit promotion workflow verification.",
    )
    assert (
        registry_client.get_model_version_by_alias(
            config.registered_model_name,
            "staging",
        ).version
        == int(reference.registered_model_version)
    )
