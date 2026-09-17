"""Typed loading and validation for Sprint 3 tracking configuration."""

from __future__ import annotations

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
    artifact_location: str


def load_tracking_config(config_path: str | Path) -> TrackingConfig:
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
    return TrackingConfig(
        uri=_require_non_empty_string(tracking_config, "uri"),
        experiment_name=_require_non_empty_string(tracking_config, "experiment_name"),
        registered_model_name=_require_non_empty_string(tracking_config, "registered_model_name"),
        artifact_location=_require_non_empty_string(tracking_config, "artifact_location"),
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
