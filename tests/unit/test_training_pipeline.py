import json

from pipelines.train_models import TrainingPipelineSummary


def test_training_pipeline_summary_is_serializable():
    summary = TrainingPipelineSummary(
        train_row_count=10,
        validation_row_count=2,
        test_row_count=2,
        feature_count=3,
        model_names=("logistic_regression", "random_forest", "xgboost"),
        production_model="xgboost",
        production_threshold=0.4,
        evaluation_report="artifacts/evaluation_report.json",
    )

    assert summary.as_dict()["model_names"] == (
        "logistic_regression",
        "random_forest",
        "xgboost",
    )
    serialized = json.dumps(summary.as_dict())
    assert json.loads(serialized)["feature_count"] == 3
