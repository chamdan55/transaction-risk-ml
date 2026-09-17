"""MLflow tracking client wrapper for Sprint 3."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from ml.tracking.config import TrackingConfig


class TrackingClientError(RuntimeError):
    """Raised when the MLflow tracking client cannot be configured."""


@dataclass
class MlflowTrackingClient:
    """Small application-owned wrapper around the MLflow Python client."""

    config: TrackingConfig
    _mlflow: ModuleType | None = None
    _experiment_id: str | None = None

    def configure(self) -> str:
        """Configure tracking URI and return the experiment ID."""

        mlflow = self._load_mlflow()
        try:
            mlflow.set_tracking_uri(self.config.uri)
            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
            if experiment is None:
                self._experiment_id = mlflow.create_experiment(
                    self.config.experiment_name,
                    artifact_location=self.config.artifact_location,
                )
            else:
                self._experiment_id = str(experiment.experiment_id)
        except Exception as exc:
            raise TrackingClientError(
                f"Unable to configure MLflow experiment: {self.config.experiment_name}"
            ) from exc
        return self._experiment_id

    def start_run(
        self,
        *,
        run_name: str | None = None,
        nested: bool = False,
    ) -> AbstractContextManager[Any]:
        """Start an MLflow run in the configured experiment."""

        experiment_id = self._experiment_id or self.configure()
        mlflow = self._load_mlflow()
        return mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            nested=nested,
        )

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to the active MLflow run."""

        if not params:
            raise TrackingClientError("params must not be empty")
        mlflow = self._load_mlflow()
        try:
            mlflow.log_params(params)
        except Exception as exc:
            raise TrackingClientError("Unable to log MLflow parameters") from exc

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log numeric metrics to the active MLflow run."""

        if not metrics:
            raise TrackingClientError("metrics must not be empty")
        mlflow = self._load_mlflow()
        try:
            mlflow.log_metrics(metrics)
        except Exception as exc:
            raise TrackingClientError("Unable to log MLflow metrics") from exc

    def log_dict(self, payload: dict[str, Any], artifact_file: str) -> None:
        """Log a JSON-compatible mapping as an MLflow artifact."""

        if not payload:
            raise TrackingClientError("payload must not be empty")
        if not artifact_file.strip():
            raise TrackingClientError("artifact_file must not be empty")
        mlflow = self._load_mlflow()
        try:
            mlflow.log_dict(payload, artifact_file)
        except Exception as exc:
            raise TrackingClientError("Unable to log MLflow dictionary artifact") from exc

    def log_model(
        self,
        model: Any,
        *,
        artifact_path: str,
        registered_model_name: str | None = None,
    ) -> Any:
        """Log a complete preprocessing-plus-estimator model artifact."""

        if not artifact_path.strip():
            raise TrackingClientError("artifact_path must not be empty")
        mlflow = self._load_mlflow()
        try:
            return mlflow.sklearn.log_model(
                model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
            )
        except Exception as exc:
            raise TrackingClientError("Unable to log MLflow model artifact") from exc

    def load_model(self, model_uri: str) -> Any:
        """Load a model artifact previously logged with the sklearn flavor."""

        if not model_uri.strip():
            raise TrackingClientError("model_uri must not be empty")
        mlflow = self._load_mlflow()
        try:
            return mlflow.sklearn.load_model(model_uri)
        except Exception as exc:
            raise TrackingClientError("Unable to load MLflow model artifact") from exc

    def _load_mlflow(self) -> ModuleType:
        if self._mlflow is None:
            try:
                import mlflow
            except ImportError as exc:
                raise TrackingClientError("MLflow is required for experiment tracking") from exc
            self._mlflow = mlflow
        return self._mlflow
