"""Model artifact and registry helpers for Sprint 3."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
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


@dataclass(frozen=True)
class PromotionRecord:
    """Auditable alias transition, including the version that can be restored."""

    registered_model_name: str
    alias: str
    previous_version: str | None
    new_version: str
    approved_by: str
    reason: str
    timestamp_utc: str


@dataclass(frozen=True)
class RollbackRecord:
    """Auditable restoration of a previously approved model version."""

    registered_model_name: str
    alias: str
    source_version: str
    restored_version: str
    approved_by: str
    reason: str
    timestamp_utc: str


def log_and_register_model(
    client: MlflowTrackingClient,
    model: Any,
    *,
    model_name: str,
    artifact_path: str = "model",
    registered_model_name: str | None = None,
    version_tags: dict[str, str] | None = None,
    signature: Any | None = None,
    input_example: Any | None = None,
) -> LoggedModel:
    """Log one complete model bundle and return its registry reference."""

    if not model_name.strip():
        raise TrackingClientError("model_name must not be empty")
    model_info = client.log_model(
        model,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
    )
    model_uri = getattr(model_info, "model_uri", None)
    if not isinstance(model_uri, str) or not model_uri:
        raise TrackingClientError("MLflow model logging returned no model_uri")
    registered_version = getattr(model_info, "registered_model_version", None)
    reference = LoggedModel(
        model_name=model_name,
        # ``name`` is the MLflow 3.x API keyword; ``artifact_path`` remains the
        # stable name of the internal registry reference field.
        artifact_path=artifact_path,
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
) -> PromotionRecord:
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
    if reference.registered_model_name != client.config.registered_model_name:
        raise TrackingClientError(
            "Promotion model name does not match the configured registry model"
        )
    model_version = client.get_model_version(version=reference.registered_model_version)
    version_tags = model_version.tags
    if getattr(model_version, "status", "READY") != "READY":
        raise TrackingClientError("Only a READY MLflow model version can be promoted")
    if version_tags.get("candidate_status") not in {"candidate", "staging", "production"}:
        raise TrackingClientError("Rejected or unreviewable model versions cannot be promoted")
    if version_tags.get("retraining.quality_gate") == "rejected":
        raise TrackingClientError("A model that failed retraining quality gates cannot be promoted")
    if version_tags.get("signature_validation") != "passed":
        raise TrackingClientError("Model version has not passed signature validation")
    if version_tags.get("serving_input_validation") != "passed":
        raise TrackingClientError("Model version has not passed serving-input validation")
    previous_version = client.get_model_alias_version(alias=stage)
    if previous_version == reference.registered_model_version:
        raise TrackingClientError(
            f"Model version {reference.registered_model_version} is already assigned to alias {stage!r}"
        )
    timestamp = datetime.now(UTC).isoformat()
    try:
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
                "promotion.alias": stage,
                "promotion.approved_by": approved_by,
                "promotion.reason": reason,
                "promotion.previous_version": previous_version or "none",
                "promotion.new_version": reference.registered_model_version,
                "promotion.timestamp_utc": timestamp,
                f"promotion.previous_version.{stage}": previous_version or "none",
                f"promotion.new_version.{stage}": reference.registered_model_version,
                f"promotion.timestamp_utc.{stage}": timestamp,
                f"promotion.approved_by.{stage}": approved_by,
                f"promotion.reason.{stage}": reason,
            },
        )
    except Exception:
        try:
            if previous_version is None:
                client.delete_model_alias(alias=stage)
            else:
                client.set_model_alias(
                    registered_model_name=reference.registered_model_name,
                    alias=stage,
                    version=previous_version,
                )
        except Exception as rollback_error:
            raise TrackingClientError(
                "Promotion metadata failed and the model alias could not be restored"
            ) from rollback_error
        raise
    return PromotionRecord(
        registered_model_name=reference.registered_model_name,
        alias=stage,
        previous_version=previous_version,
        new_version=reference.registered_model_version,
        approved_by=approved_by,
        reason=reason,
        timestamp_utc=timestamp,
    )


def rollback_registered_model(
    client: MlflowTrackingClient,
    *,
    stage: str,
    approved_by: str,
    reason: str,
    expected_current_version: str | None = None,
) -> RollbackRecord:
    """Restore the version recorded by the latest successful promotion audit."""

    if stage not in {"staging", "production"}:
        raise TrackingClientError("stage must be either 'staging' or 'production'")
    if not approved_by.strip() or not reason.strip():
        raise TrackingClientError("approved_by and reason must not be empty")
    current_version = client.get_model_alias_version(alias=stage)
    if current_version is None:
        raise TrackingClientError(f"No MLflow model is currently assigned to alias {stage!r}")
    if expected_current_version is not None and current_version != expected_current_version:
        raise TrackingClientError(
            f"Alias {stage!r} points to version {current_version}, expected "
            f"{expected_current_version}; refusing to roll back a concurrent promotion"
        )
    current_metadata = client.get_model_version(version=current_version)
    previous_version = current_metadata.tags.get(
        f"promotion.previous_version.{stage}",
        current_metadata.tags.get("promotion.previous_version"),
    )
    if not previous_version or previous_version == "none":
        raise TrackingClientError(
            f"Model version {current_version} has no previously promoted {stage} version"
        )
    previous_metadata = client.get_model_version(version=previous_version)
    if getattr(previous_metadata, "status", "READY") != "READY":
        raise TrackingClientError(
            f"Cannot restore model version {previous_version}: registry status is not READY"
        )
    timestamp = datetime.now(UTC).isoformat()
    try:
        client.set_model_alias(
            registered_model_name=client.config.registered_model_name,
            alias=stage,
            version=previous_version,
        )
        client.set_model_version_tags(
            registered_model_name=client.config.registered_model_name,
            version=current_version,
            tags={
                "rollback.alias": stage,
                "rollback.from_version": current_version,
                "rollback.to_version": previous_version,
                "rollback.approved_by": approved_by,
                "rollback.reason": reason,
                "rollback.timestamp_utc": timestamp,
            },
        )
    except Exception:
        try:
            client.set_model_alias(
                registered_model_name=client.config.registered_model_name,
                alias=stage,
                version=current_version,
            )
        except Exception as rollback_error:
            raise TrackingClientError(
                "Rollback audit failed and the model alias could not be restored"
            ) from rollback_error
        raise
    return RollbackRecord(
        registered_model_name=client.config.registered_model_name,
        alias=stage,
        source_version=current_version,
        restored_version=previous_version,
        approved_by=approved_by,
        reason=reason,
        timestamp_utc=timestamp,
    )
