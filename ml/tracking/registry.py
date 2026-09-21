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
    version_tags: dict[str, str] | None = None,
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
    reference = LoggedModel(
        model_name=model_name,
        name=artifact_path,
        model_uri=model_uri,
        registered_model_name=registered_model_name,
        registered_model_version=(
            str(registered_version) if registered_version is not None else None
        ),
    )
    if (
        reference.registered_model_name is not None
        and reference.registered_model_version is not None
        and version_tags
    ):
        client.set_model_version_tags(
            registered_model_name=reference.registered_model_name,
            version=reference.registered_model_version,
            tags=version_tags,
        )
    return reference


def load_logged_model(client: MlflowTrackingClient, reference: LoggedModel) -> Any:
    """Load a model bundle from a logged model reference."""

    return client.load_model(reference.model_uri)


def promote_registered_model(
    client: MlflowTrackingClient,
    reference: LoggedModel,
    *,
    stage: str,
    approved_by: str,
    reason: str,
) -> None:
    """Promote an already registered candidate after explicit human approval.

    Promotion deliberately is not part of training.  Calling code must supply
    the reviewer and rationale so a candidate cannot silently become staging
    or production merely because it won a validation comparison.
    """

    if stage not in {"staging", "production"}:
        raise TrackingClientError("stage must be either 'staging' or 'production'")
    if not approved_by.strip() or not reason.strip():
        raise TrackingClientError("approved_by and reason must not be empty")
    if reference.registered_model_name is None or reference.registered_model_version is None:
        raise TrackingClientError("A registered model version is required for promotion")
    client.set_model_alias(
        registered_model_name=reference.registered_model_name,
        alias=stage,
        version=reference.registered_model_version,
    )
    client.set_model_version_tags(
        registered_model_name=reference.registered_model_name,
        version=reference.registered_model_version,
        tags={
            "candidate_status": stage,
            "promotion.approved_by": approved_by,
            "promotion.reason": reason,
        },
    )
