import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from ml.tracking.client import MlflowTrackingClient, TrackingClientError
from ml.tracking.config import TrackingConfig


class FakeMlflow:
    def __init__(self):
        self.tracking_uri = None
        self.created_experiment = None
        self.started_run = None
        self.logged_params = None
        self.logged_metrics = None
        self.logged_dict = None
        self.logged_model = None

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def get_experiment_by_name(self, _name):
        return None

    def create_experiment(self, name, **kwargs):
        self.created_experiment = (name, kwargs)
        return "experiment-1"

    def log_params(self, params):
        self.logged_params = params

    def log_metrics(self, metrics):
        self.logged_metrics = metrics

    def log_dict(self, payload, artifact_file):
        self.logged_dict = (payload, artifact_file)

    class sklearn:
        logged = None

        @classmethod
        def log_model(cls, model, name, registered_model_name, serialization_format):
            cls.logged = (model, name, registered_model_name, serialization_format)
            return SimpleNamespace(
                model_uri="runs:/run-1/model",
                registered_model_version="1",
            )

        @classmethod
        def load_model(cls, model_uri):
            return {"model_uri": model_uri}

    class models:
        @staticmethod
        def infer_signature(input_example, output):
            return {"input": input_example, "output": output}

    @contextmanager
    def start_run(self, **kwargs):
        self.started_run = kwargs
        yield SimpleNamespace(info=SimpleNamespace(run_id="run-1"))


def test_tracking_client_configures_experiment_and_starts_run():

    fake_mlflow = FakeMlflow()
    config = TrackingConfig(
        uri="mlruns",
        experiment_name="transaction-risk-classification",
        registered_model_name="transaction-risk-model",
        artifact_location="mlartifacts",
    )
    client = MlflowTrackingClient(config, _mlflow=fake_mlflow)

    assert client.configure() == "experiment-1"
    assert fake_mlflow.tracking_uri == "mlruns"
    assert fake_mlflow.created_experiment == (
        "transaction-risk-classification",
        {"artifact_location": "mlartifacts"},
    )

    with client.start_run(run_name="baseline") as run:
        assert run.info.run_id == "run-1"
    assert fake_mlflow.started_run == {
        "experiment_id": "experiment-1",
        "run_name": "baseline",
        "nested": False,
    }


def test_tracking_client_leaves_remote_artifact_location_to_server():
    fake_mlflow = FakeMlflow()
    client = MlflowTrackingClient(
        TrackingConfig(
            uri="http://127.0.0.1:5000",
            experiment_name="remote-experiment",
            registered_model_name="risk-model",
        ),
        _mlflow=fake_mlflow,
    )

    assert client.configure() == "experiment-1"
    assert fake_mlflow.created_experiment == ("remote-experiment", {})


def test_tracking_client_forces_private_artifact_transfers_through_server(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "true")
    monkeypatch.setenv("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "true")
    client = MlflowTrackingClient(
        TrackingConfig(
            uri="http://127.0.0.1:5000",
            experiment_name="remote-experiment",
            registered_model_name="risk-model",
            force_proxy_artifact_transfers=True,
        ),
        _mlflow=FakeMlflow(),
    )

    client.configure()

    assert os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"] == "false"
    assert os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD"] == "false"


def test_tracking_client_wraps_remote_server_unavailable_error():
    class UnavailableMlflow(FakeMlflow):
        def get_experiment_by_name(self, _name):
            raise ConnectionError("connection refused")

    client = MlflowTrackingClient(
        TrackingConfig(
            uri="http://127.0.0.1:5055",
            experiment_name="remote-experiment",
            registered_model_name="risk-model",
        ),
        _mlflow=UnavailableMlflow(),
    )

    with pytest.raises(TrackingClientError, match="Unable to configure MLflow experiment"):
        client.configure()


def test_tracking_client_infers_signature_from_feature_only_input():
    fake_mlflow = FakeMlflow()
    client = MlflowTrackingClient(
        TrackingConfig("mlruns", "fraud", "risk-model", "mlartifacts"),
        _mlflow=fake_mlflow,
    )
    input_example = {"amount": [100.0], "transaction_type": ["PAYMENT"]}
    model = SimpleNamespace(predict_proba=lambda frame: [[0.9, 0.1]])

    signature = client.infer_signature(model, input_example)

    assert signature["input"] == input_example
    assert signature["output"] == [[0.9, 0.1]]


def test_tracking_client_preserves_signature_inference_failure_reason():
    fake_mlflow = FakeMlflow()
    client = MlflowTrackingClient(
        TrackingConfig("mlruns", "fraud", "risk-model", "mlartifacts"),
        _mlflow=fake_mlflow,
    )
    model = SimpleNamespace(
        predict_proba=lambda _frame: (_ for _ in ()).throw(ValueError("feature-only failure"))
    )

    with pytest.raises(TrackingClientError, match="feature-only failure"):
        client.infer_signature(model, {"amount": [100.0]})
