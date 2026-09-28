"""Typed loading and validation for Sprint 3 tracking configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class TrackingConfigurationError(ValueError):
    """Raised when tracking configuration is missing or invalid."""


@dataclass(frozen=True)
class TrackingConfig:
    """Validated configuration for the MLflow tracking layer."""

    uri: str
    experiment_name: str
    registered_model_name: str
    artifact_location: str | None = None
    force_proxy_artifact_transfers: bool = False


def configure_artifact_transfers(config: TrackingConfig) -> None:
    """Keep clients off a private artifact store when the server proxies it."""

    if config.force_proxy_artifact_transfers:
        os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"] = "false"
        os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD"] = "false"


def load_tracking_config(
    config_path: str | Path,
    *,
    tracking_uri: str | None = None,
) -> TrackingConfig:
    """Load and validate a tracking configuration YAML file."""

    path = Path(config_path)
    try:
        with path.open(encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)
    except OSError as exc:
        raise TrackingConfigurationError(
            f"Unable to read tracking configuration from {path}"
        ) from exc
    except yaml.YAMLError as exc:
        raise TrackingConfigurationError(f"Invalid YAML in tracking configuration: {path}") from exc

    if not isinstance(raw_config, dict):
        raise TrackingConfigurationError("Tracking configuration must be a mapping")
    tracking_config = _require_mapping(raw_config, "tracking")
    configured_uri = _require_non_empty_string(tracking_config, "uri")
    if tracking_uri is not None:
        if not tracking_uri.strip():
            raise TrackingConfigurationError("tracking_uri override must not be empty")
        configured_uri = tracking_uri.strip()

    artifact_location = tracking_config.get("artifact_location")
    if artifact_location is not None and (
        not isinstance(artifact_location, str) or not artifact_location.strip()
    ):
        raise TrackingConfigurationError(
            "artifact_location must be a non-empty string or null when provided"
        )

    force_proxy_artifact_transfers = tracking_config.get("force_proxy_artifact_transfers", False)
    if not isinstance(force_proxy_artifact_transfers, bool):
        raise TrackingConfigurationError("force_proxy_artifact_transfers must be a boolean")

    return TrackingConfig(
        uri=configured_uri,
        experiment_name=_require_non_empty_string(tracking_config, "experiment_name"),
        registered_model_name=_require_non_empty_string(tracking_config, "registered_model_name"),
        artifact_location=artifact_location.strip() if artifact_location else None,
        force_proxy_artifact_transfers=force_proxy_artifact_transfers,
    )


def _require_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise TrackingConfigurationError(f"{key} must be a mapping")
    return value


def _require_non_empty_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TrackingConfigurationError(f"{key} must be a non-empty string")
    return value
