from types import SimpleNamespace

from ml.tracking.client import MlflowTrackingClient
from ml.tracking.config import TrackingConfig
from ml.tracking.registry import load_logged_model, log_and_register_model


class FakeSklearn:
    logged = None

    @classmethod
    def log_model(cls, model, artifact_path, registered_model_name):
        cls.logged = (model, artifact_path, registered_model_name)
        return SimpleNamespace(
            model_uri="runs:/run-1/model",
            registered_model_version="1",
        )

    @classmethod
    def load_model(cls, model_uri):
        return {"model_uri": model_uri}


class FakeMlflow:
    sklearn = FakeSklearn


def test_model_artifact_is_logged_registered_and_loadable():
    client = MlflowTrackingClient(
        TrackingConfig("mlruns", "fraud", "risk-model", "mlartifacts"),
        _mlflow=FakeMlflow(),
    )
    model = object()

    reference = log_and_register_model(
        client,
        model,
        model_name="xgboost",
        artifact_path="models/xgboost",
        registered_model_name="transaction-risk-model",
    )
    loaded = load_logged_model(client, reference)

    assert reference.model_uri == "runs:/run-1/model"
    assert reference.registered_model_version == "1"
    assert loaded == {"model_uri": "runs:/run-1/model"}
    assert FakeSklearn.logged == (
        model,
        "models/xgboost",
        "transaction-risk-model",
    )
