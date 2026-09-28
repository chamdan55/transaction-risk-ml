"""Validate local secrets required by the optional containerized MLflow profile."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path


class TrackingEnvironmentError(ValueError):
    """Raised when the local tracking profile environment is incomplete or unsafe."""


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def validate_tracking_environment(
    *,
    env_file: str | Path = ".env",
    process_env: Mapping[str, str] | None = None,
) -> None:
    """Require local credentials and URL-safe values without exposing their contents."""

    environment = os.environ if process_env is None else process_env
    file_values = _read_env_file(Path(env_file))
    values = {
        key: environment.get(key, file_values.get(key, ""))
        for key in ("POSTGRES_PASSWORD", "MINIO_ROOT_USER", "MINIO_ROOT_PASSWORD")
    }

    missing = [key for key, value in values.items() if not value]
    if missing:
        raise TrackingEnvironmentError(
            "Set these tracking-profile values in .env or the process environment: "
            + ", ".join(missing)
        )
    if not re.fullmatch(r"[A-Fa-f0-9]{32,128}", values["POSTGRES_PASSWORD"]):
        raise TrackingEnvironmentError(
            "POSTGRES_PASSWORD must be 32-128 hexadecimal characters so the database URL is safe"
        )
    if len(values["MINIO_ROOT_USER"]) < 3:
        raise TrackingEnvironmentError("MINIO_ROOT_USER must contain at least 3 characters")
    if len(values["MINIO_ROOT_PASSWORD"]) < 8:
        raise TrackingEnvironmentError("MINIO_ROOT_PASSWORD must contain at least 8 characters")

    port = environment.get("MLFLOW_PORT", file_values.get("MLFLOW_PORT", "5000"))
    try:
        numeric_port = int(port)
    except ValueError as exc:
        raise TrackingEnvironmentError(
            "MLFLOW_PORT must be an integer between 1 and 65535"
        ) from exc
    if not 1 <= numeric_port <= 65535:
        raise TrackingEnvironmentError("MLFLOW_PORT must be an integer between 1 and 65535")


def main() -> None:
    try:
        validate_tracking_environment()
    except TrackingEnvironmentError as exc:
        raise SystemExit(str(exc)) from exc
    print("MLflow tracking environment is valid; secret values were not displayed.")


if __name__ == "__main__":
    main()
