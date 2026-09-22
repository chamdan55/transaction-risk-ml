"""Lifespan-managed model loading and prediction service."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from math import isfinite
from typing import Any

import joblib
import pandas as pd

from app.core.config import Settings
from ml.contracts.features import FEATURE_CONTRACT_VERSION, validate_model_feature_columns

LOGGER = logging.getLogger(__name__)


class ModelProviderError(RuntimeError):
    """Raised when a model cannot safely be made available for serving."""


@dataclass(frozen=True)
class ModelMetadata:
    model_name: str
    model_version: str
    model_source: str
    threshold: float
    threshold_policy: str
    feature_contract_version: str


class ModelProvider:
    """Load exactly once at startup and retain a validated model in memory."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any | None = None
        self._metadata: ModelMetadata | None = None
        self._load_error: str | None = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._metadata is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def metadata(self) -> ModelMetadata:
        if self._metadata is None:
            raise ModelProviderError("Model is not ready")
        return self._metadata

    def load(self) -> None:
        """Load and validate the configured model without taking down liveness."""

        self._model = None
        self._metadata = None
        self._load_error = None
        try:
            if self._settings.model_source == "local":
                model, metadata = self._load_local()
            else:
                model, metadata = self._load_mlflow_alias()
            self._validate_model(model, metadata)
        except Exception as exc:
            self._load_error = str(exc)
            LOGGER.exception("Model provider is not ready: %s", exc)
            return
        self._model = model
        self._metadata = metadata
        LOGGER.info(
            "Model provider ready: source=%s model=%s version=%s contract=%s",
            metadata.model_source,
            metadata.model_name,
            metadata.model_version,
            metadata.feature_contract_version,
        )

    def predict_score(self, feature_frame: pd.DataFrame) -> float:
        """Score one already-prepared frame; loading never occurs on this path."""

        if not self.is_ready:
            raise ModelProviderError("Model is not ready")
        probabilities = self._model.predict_proba(feature_frame)
        try:
            score = float(probabilities[0][1])
        except (IndexError, KeyError, TypeError) as exc:
            raise ModelProviderError("Model returned an invalid probability shape") from exc
        if not isfinite(score) or not 0 <= score <= 1:
            raise ModelProviderError("Model returned a probability outside [0, 1]")
        return score

    def _load_local(self) -> tuple[Any, ModelMetadata]:
        artifact_path = self._settings.model_artifact_path
        report_path = self._settings.model_evaluation_report_path
        if not artifact_path.is_file():
            raise ModelProviderError(f"Local model artifact does not exist: {artifact_path}")
        if not report_path.is_file():
            raise ModelProviderError(f"Local evaluation report does not exist: {report_path}")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelProviderError("Local evaluation report is not valid JSON") from exc
        model = joblib.load(artifact_path)
        candidate = _mapping(report, "candidate")
        config = _mapping(report, "config")
        lineage = _mapping(report, "lineage")
        expected_artifact_name = f"{candidate['model_name']}.joblib"
        if artifact_path.name != expected_artifact_name:
            raise ModelProviderError(
                "Local artifact does not match evaluation candidate: "
                f"expected {expected_artifact_name}, got {artifact_path.name}"
            )
        return model, _metadata_from_values(
            model_name=candidate["model_name"],
            model_version=f"local-{_required_string(lineage, 'manifest_id')[:12]}",
            model_source="local",
            threshold=candidate["threshold"],
            threshold_policy=candidate["threshold_selection_strategy"],
            feature_contract_version=config["feature_contract_version"],
        )

    def _load_mlflow_alias(self) -> tuple[Any, ModelMetadata]:
        try:
            import mlflow
        except ImportError as exc:
            raise ModelProviderError("MLflow is required for model_source=mlflow_alias") from exc
        try:
            mlflow.set_tracking_uri(self._settings.mlflow_tracking_uri)
            client = mlflow.tracking.MlflowClient(tracking_uri=self._settings.mlflow_tracking_uri)
            version = client.get_model_version_by_alias(
                self._settings.model_name, self._settings.model_registry_alias
            )
            model = mlflow.sklearn.load_model(
                f"models:/{self._settings.model_name}@{self._settings.model_registry_alias}"
            )
        except Exception as exc:
            raise ModelProviderError(
                "Unable to load the configured MLflow model alias "
                f"{self._settings.model_name}@{self._settings.model_registry_alias}"
            ) from exc
        tags = version.tags
        if tags.get("signature_validation") != "passed":
            raise ModelProviderError("MLflow model version has not passed signature validation")
        if tags.get("serving_input_validation") != "passed":
            raise ModelProviderError("MLflow model version has not passed serving input validation")
        return model, _metadata_from_values(
            model_name=tags.get("model_name", self._settings.model_name),
            model_version=str(version.version),
            model_source="mlflow_alias",
            threshold=tags.get("production_threshold"),
            threshold_policy=tags.get("threshold_selection_strategy"),
            feature_contract_version=tags.get("feature_contract_version"),
        )

    @staticmethod
    def _validate_model(model: Any, metadata: ModelMetadata) -> None:
        if metadata.feature_contract_version != FEATURE_CONTRACT_VERSION:
            raise ModelProviderError(
                "Model feature contract version does not match serving: "
                f"{metadata.feature_contract_version!r}"
            )
        preprocessor = getattr(model, "preprocessor", None)
        feature_columns = getattr(preprocessor, "feature_columns", None)
        try:
            validate_model_feature_columns(feature_columns)
        except (TypeError, ValueError) as exc:
            raise ModelProviderError(
                "Model feature columns do not match the serving contract"
            ) from exc
        if not callable(getattr(model, "predict_proba", None)):
            raise ModelProviderError("Model does not expose predict_proba")


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ModelProviderError(f"Evaluation report is missing object: {key}")
    return value


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ModelProviderError(f"Required metadata is missing or invalid: {key}")
    return value


def _metadata_from_values(
    *,
    model_name: Any,
    model_version: Any,
    model_source: str,
    threshold: Any,
    threshold_policy: Any,
    feature_contract_version: Any,
) -> ModelMetadata:
    try:
        normalized_threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ModelProviderError("Model threshold is missing or invalid") from exc
    if not isfinite(normalized_threshold) or not 0 <= normalized_threshold <= 1:
        raise ModelProviderError("Model threshold must be finite and within [0, 1]")
    return ModelMetadata(
        model_name=_required_string({"value": model_name}, "value"),
        model_version=_required_string({"value": model_version}, "value"),
        model_source=model_source,
        threshold=normalized_threshold,
        threshold_policy=_required_string({"value": threshold_policy}, "value"),
        feature_contract_version=_required_string({"value": feature_contract_version}, "value"),
    )
