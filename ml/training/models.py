"""Comparison model training for Sprint 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.training.config import ModelConfig
from ml.training.imbalance import (
    TargetDistribution,
    balanced_class_weight,
    scale_pos_weight,
    summarize_target,
)
from ml.training.preprocessing import FeaturePreprocessor


class ComparisonModelError(RuntimeError):
    """Raised when a comparison model cannot be trained."""


@dataclass
class TrainedComparisonModel:
    """Fitted preprocessing and estimator for a comparison model."""

    model_name: str
    estimator: Any
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


def train_random_forest(
    train_frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    config: ModelConfig,
) -> TrainedComparisonModel:
    """Train the Random Forest comparison model."""

    target = _require_training_target(train_frame, config.target_column)
    target_distribution = summarize_target(target)
    preprocessor = FeaturePreprocessor(feature_columns, config.target_column)
    transformed_features = preprocessor.fit_transform(train_frame)
    model_params = _prepare_params(
        config.model_params["random_forest"],
        target_distribution=target_distribution,
        random_seed=config.random_seed,
    )
    estimator = RandomForestClassifier(**model_params)
    estimator.fit(transformed_features, target)
    return TrainedComparisonModel(
        model_name="random_forest",
        estimator=estimator,
        preprocessor=preprocessor,
        target_distribution=target_distribution,
        training_row_count=len(train_frame),
    )


def train_xgboost(
    train_frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    config: ModelConfig,
) -> TrainedComparisonModel:
    """Train the XGBoost comparison model."""

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ComparisonModelError(
            "XGBoost is required to train the xgboost comparison model"
        ) from exc

    target = _require_training_target(train_frame, config.target_column)
    target_distribution = summarize_target(target)
    preprocessor = FeaturePreprocessor(feature_columns, config.target_column)
    transformed_features = preprocessor.fit_transform(train_frame)
    model_params = dict(config.model_params["xgboost"])
    model_params.setdefault("scale_pos_weight", scale_pos_weight(target_distribution))
    model_params.setdefault("random_state", config.random_seed)
    model_params.setdefault("n_jobs", -1)
    estimator = XGBClassifier(**model_params)
    estimator.fit(transformed_features, target)
    return TrainedComparisonModel(
        model_name="xgboost",
        estimator=estimator,
        preprocessor=preprocessor,
        target_distribution=target_distribution,
        training_row_count=len(train_frame),
    )


def _prepare_params(
    params: dict[str, Any],
    *,
    target_distribution: TargetDistribution,
    random_seed: int,
) -> dict[str, Any]:
    prepared_params = dict(params)
    if prepared_params.get("class_weight") == "balanced":
        prepared_params["class_weight"] = balanced_class_weight(target_distribution)
    prepared_params.setdefault("random_state", random_seed)
    prepared_params.setdefault("n_jobs", -1)
    return prepared_params


def _require_training_target(train_frame: pd.DataFrame, target_column: str) -> pd.Series:
    if target_column not in train_frame.columns:
        raise ValueError(f"Training target column not found: {target_column}")
    return train_frame[target_column]
