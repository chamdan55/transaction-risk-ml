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

    def set_tags(self, tags: dict[str, str]) -> None:
        """Attach stable, queryable tags to the active MLflow run."""

        if not tags:
            raise TrackingClientError("tags must not be empty")
        try:
            self._load_mlflow().set_tags(tags)
        except Exception as exc:
            raise TrackingClientError("Unable to set MLflow run tags") from exc

    def log_model(
        self,
        model: Any,
        *,
        artifact_path: str,
        registered_model_name: str | None = None,
        signature: Any | None = None,
        input_example: Any | None = None,
    ) -> Any:
        """Log a complete preprocessing-plus-estimator model artifact."""

        if not artifact_path.strip():
            raise TrackingClientError("artifact_path must not be empty")
        mlflow = self._load_mlflow()
        try:
            kwargs = {
                "name": artifact_path,
                "registered_model_name": registered_model_name,
                "serialization_format": "cloudpickle",
            }
            if signature is not None:
                kwargs["signature"] = signature
            if input_example is not None:
                kwargs["input_example"] = input_example
            return mlflow.sklearn.log_model(model, **kwargs)
        except Exception as exc:
            raise TrackingClientError("Unable to log MLflow model artifact") from exc

    def infer_signature(self, model: Any, input_example: Any) -> Any:
        """Infer an MLflow model signature from the serving input example."""

        mlflow = self._load_mlflow()
        try:
            return mlflow.models.infer_signature(
                input_example,
                model.predict_proba(input_example),
            )
        except Exception as exc:
            raise TrackingClientError(f"Unable to infer MLflow model signature: {exc}") from exc

    def load_model(self, model_uri: str) -> Any:
        """Load a model artifact previously logged with the sklearn flavor."""

        if not model_uri.strip():
            raise TrackingClientError("model_uri must not be empty")
        mlflow = self._load_mlflow()
        try:
            return mlflow.sklearn.load_model(model_uri)
        except Exception as exc:
            raise TrackingClientError("Unable to load MLflow model artifact") from exc

    def set_model_version_tags(
        self,
        *,
        registered_model_name: str,
        version: str,
        tags: dict[str, str],
    ) -> None:
        """Persist traceability metadata on one registered model version."""

        if not registered_model_name.strip() or not version.strip() or not tags:
            raise TrackingClientError("registered model name, version, and tags are required")
        try:
            registry_client = self._load_mlflow().tracking.MlflowClient(
                tracking_uri=self.config.uri
            )
            for key, value in tags.items():
                registry_client.set_model_version_tag(
                    registered_model_name,
                    version,
                    key,
                    value,
                )
        except Exception as exc:
            raise TrackingClientError("Unable to set MLflow model version tags") from exc

    def set_model_alias(
        self,
        *,
        registered_model_name: str,
        alias: str,
        version: str,
    ) -> None:
        """Point a registered-model alias at an explicitly approved version."""

        if not registered_model_name.strip() or not alias.strip() or not version.strip():
            raise TrackingClientError("registered model name, alias, and version are required")
        try:
            self._load_mlflow().tracking.MlflowClient(
                tracking_uri=self.config.uri
            ).set_registered_model_alias(registered_model_name, alias, version)
        except Exception as exc:
            raise TrackingClientError("Unable to set MLflow model alias") from exc

    def _load_mlflow(self) -> ModuleType:
        if self._mlflow is None:
            try:
                import mlflow
            except ImportError as exc:
                raise TrackingClientError("MLflow is required for experiment tracking") from exc
            self._mlflow = mlflow
        return self._mlflow
