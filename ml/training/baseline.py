"""Logistic Regression baseline training for Sprint 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.training.config import ModelConfig
from ml.training.imbalance import (
    TargetDistribution,
    balanced_class_weight,
    summarize_target,
)
from ml.training.preprocessing import FeaturePreprocessor


@dataclass
class LogisticRegressionBaseline:
    """Fitted preprocessing and Logistic Regression baseline."""

    estimator: LogisticRegression
    preprocessor: FeaturePreprocessor
    target_distribution: TargetDistribution
    training_row_count: int

    def predict_proba(self, frame: pd.DataFrame):
        """Return class probabilities for a new feature frame."""

        transformed_features = self.preprocessor.transform(frame)
        return self.estimator.predict_proba(transformed_features)

    def predict(self, frame: pd.DataFrame):
        """Return binary predictions for a new feature frame."""

        transformed_features = self.preprocessor.transform(frame)
        return self.estimator.predict(transformed_features)


def train_logistic_regression_baseline(
    train_frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    config: ModelConfig,
) -> LogisticRegressionBaseline:
    """Fit the Logistic Regression baseline on the training frame only."""

    target = _require_training_target(train_frame, config.target_column)
    target_distribution = summarize_target(target)
    preprocessor = FeaturePreprocessor(feature_columns, config.target_column)
    transformed_features = preprocessor.fit_transform(train_frame)

    model_params = dict(config.model_params["logistic_regression"])
    if model_params.get("class_weight") == "balanced":
        model_params["class_weight"] = balanced_class_weight(target_distribution)
    model_params.setdefault("random_state", config.random_seed)

    estimator = LogisticRegression(**model_params)
    estimator.fit(transformed_features, target)
    return LogisticRegressionBaseline(
        estimator=estimator,
        preprocessor=preprocessor,
        target_distribution=target_distribution,
        training_row_count=len(train_frame),
    )


def save_logistic_regression_baseline(
    baseline: LogisticRegressionBaseline,
    output_path: str | Path,
) -> Path:
    """Persist the fitted baseline and preprocessing state as one artifact."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline, path)
    return path


def load_logistic_regression_baseline(
    artifact_path: str | Path,
) -> LogisticRegressionBaseline:
    """Load a persisted Logistic Regression baseline artifact."""

    artifact = joblib.load(Path(artifact_path))
    if not isinstance(artifact, LogisticRegressionBaseline):
        raise TypeError("Artifact is not a LogisticRegressionBaseline")
    return artifact


def _require_training_target(train_frame: pd.DataFrame, target_column: str) -> pd.Series:
    if target_column not in train_frame.columns:
        raise ValueError(f"Training target column not found: {target_column}")
    return train_frame[target_column]
