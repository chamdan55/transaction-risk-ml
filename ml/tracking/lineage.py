"""Stable dataset, schema, configuration, and code lineage manifests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetManifest:
    """Reproducibility identity for one training dataset and feature contract."""

    manifest_version: str
    manifest_id: str
    dataset_name: str
    dataset_path: str
    target_column: str
    feature_columns: tuple[str, ...]
    feature_contract_version: str
    schema_fingerprint: str
    content_fingerprint: str
    config_fingerprint: str
    code_fingerprint: str
    split_row_counts: dict[str, int]
    time_ranges: dict[str, dict[str, str | None]]
    generated_at: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def stable_dict(self) -> dict[str, Any]:
        """Return manifest fields suitable for reproducible reports."""

        payload = self.as_dict()
        payload.pop("generated_at", None)
        return payload


def build_dataset_manifest(
    *,
    dataset_summary: dict[str, Any],
    dataset_path: str | Path,
    config_path: str | Path,
    dataset_name: str = "paysim",
    target_column: str,
    feature_contract_version: str,
    repository_root: str | Path | None = None,
) -> DatasetManifest:
    """Build a deterministic manifest and a non-deterministic generation timestamp.

    The content fingerprint combines the validated split summary with SHA-256 hashes of the
    persisted dataset files. The timestamp is metadata only and is deliberately excluded from the
    manifest ID so repeated training runs over identical inputs remain comparable.
    """

    root = Path(dataset_path)
    config = Path(config_path)
    split_summaries = dataset_summary.get("split_summaries")
    feature_columns = dataset_summary.get("feature_columns")
    if not isinstance(split_summaries, dict) or not isinstance(feature_columns, list | tuple):
        raise ValueError("dataset_summary must contain split_summaries and feature_columns")
    if (
        not dataset_name.strip()
        or not target_column.strip()
        or not feature_contract_version.strip()
    ):
        raise ValueError("dataset_name, target_column, and feature_contract_version are required")

    split_row_counts: dict[str, int] = {}
    time_ranges: dict[str, dict[str, str | None]] = {}
    for split_name in ("train", "validation", "test"):
        summary = split_summaries.get(split_name)
        if not isinstance(summary, dict) or not isinstance(summary.get("row_count"), int):
            raise ValueError(f"dataset_summary is missing {split_name}.row_count")
        split_row_counts[split_name] = summary["row_count"]
        time_ranges[split_name] = {
            "min": _optional_string(summary.get("min_timestamp")),
            "max": _optional_string(summary.get("max_timestamp")),
        }

    schema_payload = {
        "feature_columns": list(feature_columns),
        "target_column": target_column,
        "feature_contract_version": feature_contract_version,
    }
    file_manifest = _file_manifest(root)
    content_payload = {"dataset_summary": dataset_summary, "files": file_manifest}
    config_fingerprint = _hash_file(config)
    code_fingerprint = _hash_code(repository_root or config.parent)
    stable_payload = {
        "manifest_version": "lineage-v1",
        "dataset_name": dataset_name,
        "dataset_path": str(root),
        "schema": schema_payload,
        "content_fingerprint": _hash_payload(content_payload),
        "config_fingerprint": config_fingerprint,
        "code_fingerprint": code_fingerprint,
        "split_row_counts": split_row_counts,
        "time_ranges": time_ranges,
    }
    manifest_id = _hash_payload(stable_payload)
    return DatasetManifest(
        manifest_version="lineage-v1",
        manifest_id=manifest_id,
        dataset_name=dataset_name,
        dataset_path=str(root),
        target_column=target_column,
        feature_columns=tuple(feature_columns),
        feature_contract_version=feature_contract_version,
        schema_fingerprint=_hash_payload(schema_payload),
        content_fingerprint=stable_payload["content_fingerprint"],
        config_fingerprint=config_fingerprint,
        code_fingerprint=code_fingerprint,
        split_row_counts=split_row_counts,
        time_ranges=time_ranges,
        generated_at=datetime.now(UTC).isoformat(),
    )


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _hash_payload(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    if not path.is_file():
        return _hash_payload({"missing_file": str(path)})
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_manifest(root: Path) -> list[dict[str, Any]]:
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = sorted(path for path in root.rglob("*") if path.is_file())
    else:
        return []
    return [
        {
            "path": str(path.relative_to(root.parent if root.is_file() else root)),
            "size": path.stat().st_size,
            "sha256": _hash_file(path),
        }
        for path in files
    ]


def _hash_code(repository_root: str | Path) -> str:
    root = Path(repository_root)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    source_files = []
    for directory in (root / "ml", root / "pipelines"):
        if directory.is_dir():
            source_files.extend(path for path in directory.rglob("*.py") if path.is_file())
    return _hash_payload(
        {
            "git_commit": commit,
            "files": [
                {
                    "path": str(path.relative_to(root)),
                    "sha256": _hash_file(path),
                }
                for path in sorted(source_files)
            ],
        }
    )
