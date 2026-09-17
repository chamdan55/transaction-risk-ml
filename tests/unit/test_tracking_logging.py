from ml.evaluation.threshold import analyze_thresholds, select_best_threshold
from ml.tracking.client import MlflowTrackingClient
from ml.tracking.config import TrackingConfig
from ml.tracking.logging import (
    flatten_parameters,
    log_training_parameters,
    log_validation_metrics,
)


class FakeMlflow:
    def __init__(self):
        self.logged_params = None
        self.logged_metrics = None
        self.logged_dict = None

    def log_params(self, params):
        self.logged_params = params

    def log_metrics(self, metrics):
        self.logged_metrics = metrics

    def log_dict(self, payload, artifact_file):
        self.logged_dict = (payload, artifact_file)


def test_flatten_parameters_creates_stable_names():
    assert flatten_parameters(
        {"model": {"max_depth": 6}, "seed": 42},
    ) == {"model.max_depth": 6, "seed": 42}


def test_training_parameters_are_logged_for_one_model_run():
    fake_mlflow = FakeMlflow()
    client = MlflowTrackingClient(
        TrackingConfig("mlruns", "fraud", "risk-model", "mlartifacts"),
        _mlflow=fake_mlflow,
    )

    logged = log_training_parameters(
        client,
        model_name="xgboost",
        config=TrackingConfig("mlruns", "fraud", "risk-model", "mlartifacts"),
        training_config={
            "random_seed": 42,
            "model": {"max_depth": 6, "learning_rate": 0.1},
        },
        dataset_summary={"feature_count": 10, "train_row_count": 100},
    )

    assert logged["model_name"] == "xgboost"
    assert logged["training.model.max_depth"] == 6
    assert logged["dataset.feature_count"] == 10
    assert fake_mlflow.logged_params == logged


def test_validation_metrics_and_threshold_analysis_are_logged():
    fake_mlflow = FakeMlflow()
    client = MlflowTrackingClient(
        TrackingConfig("mlruns", "fraud", "risk-model", "mlartifacts"),
        _mlflow=fake_mlflow,
    )
    evaluations = analyze_thresholds(
        [0, 0, 1, 1],
        [0.05, 0.40, 0.60, 0.90],
        [0.5, 0.8],
    )
    selected = select_best_threshold(evaluations)

    metrics = log_validation_metrics(
        client,
        model_name="xgboost",
        selected_evaluation=selected,
        threshold_evaluations=evaluations,
    )

    assert metrics["validation.pr_auc"] == 1.0
    assert metrics["validation.selected_threshold"] == 0.5
    assert fake_mlflow.logged_metrics == metrics
    assert fake_mlflow.logged_dict[0]["model_name"] == "xgboost"
    assert fake_mlflow.logged_dict[1] == "threshold_analysis/xgboost.json"
