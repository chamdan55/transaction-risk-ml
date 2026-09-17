"""Dataset and run metadata helpers for Sprint 3."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml.tracking.client import TrackingClientError


@dataclass(frozen=True)
class RunMetadata:
    """Traceability metadata attached to one training run."""

    dataset_name: str
    target_column: str
    feature_count: int
    train_row_count: int
    validation_row_count: int
    test_row_count: int
    config_hash: str
    git_commit: str
    training_timestamp: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON- and MLflow-compatible mapping."""

        return asdict(self)


def build_run_metadata(
    *,
    dataset_summary: dict[str, Any],
    target_column: str,
    config_path: str | Path,
    dataset_name: str = "paysim",
    repository_root: str | Path | None = None,
) -> RunMetadata:
    """Build traceability metadata from the training contract and config."""

    if not dataset_name.strip():
        raise TrackingClientError("dataset_name must not be empty")
    if not target_column.strip():
        raise TrackingClientError("target_column must not be empty")
    split_summaries = dataset_summary.get("split_summaries")
    if not isinstance(split_summaries, dict):
        raise TrackingClientError("dataset_summary must contain split_summaries")
    row_counts = {
        split_name: _require_row_count(split_summaries, split_name)
        for split_name in ("train", "validation", "test")
    }
    feature_columns = dataset_summary.get("feature_columns")
    if not isinstance(feature_columns, list | tuple) or not feature_columns:
        raise TrackingClientError("dataset_summary must contain non-empty feature_columns")

    config_file = Path(config_path)
    if not config_file.is_file():
        raise TrackingClientError(f"Configuration file not found: {config_file}")
    return RunMetadata(
        dataset_name=dataset_name,
        target_column=target_column,
        feature_count=len(feature_columns),
        train_row_count=row_counts["train"],
        validation_row_count=row_counts["validation"],
        test_row_count=row_counts["test"],
        config_hash=_hash_file(config_file),
        git_commit=_get_git_commit(repository_root or config_file.parent),
        training_timestamp=datetime.now(UTC).isoformat(),
    )


def _require_row_count(split_summaries: dict[str, Any], split_name: str) -> int:
    summary = split_summaries.get(split_name)
    if not isinstance(summary, dict) or not isinstance(summary.get("row_count"), int):
        raise TrackingClientError(f"dataset_summary is missing {split_name}.row_count")
    return summary["row_count"]


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _get_git_commit(repository_root: str | Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"
