"""Model artifact and registry helpers for Sprint 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ml.tracking.client import MlflowTrackingClient, TrackingClientError


@dataclass(frozen=True)
class LoggedModel:
    """Reference to a logged and optionally registered model artifact."""

    model_name: str
    artifact_path: str
    model_uri: str
    registered_model_name: str | None
    registered_model_version: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "artifact_path": self.artifact_path,
            "model_uri": self.model_uri,
            "registered_model_name": self.registered_model_name,
            "registered_model_version": self.registered_model_version,
        }


def log_and_register_model(
    client: MlflowTrackingClient,
    model: Any,
    *,
    model_name: str,
    artifact_path: str = "model",
    registered_model_name: str | None = None,
) -> LoggedModel:
    """Log one complete model bundle and return its registry reference."""

    if not model_name.strip():
        raise TrackingClientError("model_name must not be empty")
    model_info = client.log_model(
        model,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name,
    )
    model_uri = getattr(model_info, "model_uri", None)
    if not isinstance(model_uri, str) or not model_uri:
        raise TrackingClientError("MLflow model logging returned no model_uri")
    registered_version = getattr(model_info, "registered_model_version", None)
    return LoggedModel(
        model_name=model_name,
        artifact_path=artifact_path,
        model_uri=model_uri,
        registered_model_name=registered_model_name,
        registered_model_version=(
            str(registered_version) if registered_version is not None else None
        ),
    )


def load_logged_model(client: MlflowTrackingClient, reference: LoggedModel) -> Any:
    """Load a model bundle from a logged model reference."""

    return client.load_model(reference.model_uri)
