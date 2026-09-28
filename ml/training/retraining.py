"""Approval manifests and deterministic quality gates for controlled retraining."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


class RetrainingError(RuntimeError):
    """Raised when a retraining manifest, policy, or candidate is invalid."""


@dataclass(frozen=True)
class RetrainingQualityPolicy:
    """Configured absolute candidate limits and non-regression allowances."""

    absolute: dict[str, float]
    minimum_test_rows: int
    minimum_test_positives: int
    comparison: dict[str, float]
    policy_sha256: str


def source_root() -> Path:
    """Return the source package root in both a checkout and an installed wheel."""

    return Path(__file__).resolve().parents[2]


def create_approved_manifest(
    manifest: dict[str, Any], *, approved_by: str, reason: str
) -> dict[str, Any]:
    """Wrap a content-addressed dataset manifest in an explicit approval record."""

    if not approved_by.strip() or not reason.strip():
        raise RetrainingError("approved_by and approval reason are required")
    if not manifest.get("manifest_id"):
        raise RetrainingError("dataset manifest has no manifest_id")
    payload = {
        "format_version": "approved-dataset-manifest-v1",
        "manifest": manifest,
        "approval": {
            "approved_by": approved_by.strip(),
            "reason": reason.strip(),
            "approved_at": datetime.now(UTC).isoformat(),
        },
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return payload


def write_approved_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    """Write an approved manifest exactly once so a prior approval is never overwritten."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8", newline="\n") as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
    except FileExistsError as exc:
        raise RetrainingError(
            f"Refusing to overwrite approved manifest; choose a new path: {output}"
        ) from exc
    except OSError as exc:
        raise RetrainingError(f"Unable to write approved manifest: {output}") from exc


def load_approved_manifest(path: str | Path) -> dict[str, Any]:
    """Read an approved manifest and verify its content digest and approval metadata."""

    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrainingError(f"Unable to read valid approved manifest: {manifest_path}") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != (
        "approved-dataset-manifest-v1"
    ):
        raise RetrainingError("Unsupported approved dataset manifest format")
    digest = payload.get("manifest_sha256")
    unsigned_payload = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if not isinstance(digest, str) or digest != _canonical_sha256(unsigned_payload):
        raise RetrainingError("Approved dataset manifest digest does not match its contents")
    approved_manifest = payload.get("manifest")
    approval = payload.get("approval")
    if not isinstance(approved_manifest, dict) or not approved_manifest.get("manifest_id"):
        raise RetrainingError("Approved dataset manifest is missing its dataset identity")
    if not isinstance(approval, dict) or not all(
        isinstance(approval.get(key), str) and approval[key].strip()
        for key in ("approved_by", "reason", "approved_at")
    ):
        raise RetrainingError("Approved dataset manifest is missing approval metadata")
    return payload


def validate_manifest_matches(approved: dict[str, Any], current_manifest: dict[str, Any]) -> None:
    """Fail closed when dataset bytes, split/schema, config, or training code changed."""

    # Approved manifests have passed through JSON, which turns tuples (notably
    # feature_columns) into lists. Compare the same on-disk representation.
    expected = json.loads(json.dumps(approved["manifest"], sort_keys=True))
    expected.pop("generated_at", None)
    current = json.loads(json.dumps(current_manifest, sort_keys=True))
    current.pop("generated_at", None)
    if expected != current:
        changed = sorted(
            key for key in set(expected) | set(current) if expected.get(key) != current.get(key)
        )
        raise RetrainingError(
            "Current data/config/code do not match the approved dataset manifest; "
            f"changed fields: {changed}"
        )


def load_quality_policy(path: str | Path) -> RetrainingQualityPolicy:
    """Load and validate fail-closed retraining gates from YAML."""

    policy_path = Path(path)
    try:
        policy_bytes = policy_path.read_bytes()
        payload = yaml.safe_load(policy_bytes.decode("utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RetrainingError(f"Unable to read retraining policy: {policy_path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("quality_gates"), dict):
        raise RetrainingError("retraining policy must contain a quality_gates mapping")
    gates = payload["quality_gates"]
    absolute = _require_number_mapping(
        gates.get("absolute"),
        (
            "minimum_validation_pr_auc",
            "minimum_validation_precision",
            "minimum_validation_recall",
            "minimum_test_pr_auc",
            "minimum_test_precision",
            "minimum_test_recall",
            "maximum_test_expected_cost_per_transaction",
            "maximum_test_brier_score",
            "maximum_test_calibration_error",
            "maximum_test_alert_rate",
        ),
        "quality_gates.absolute",
    )
    comparison = _require_number_mapping(
        gates.get("comparison"),
        (
            "minimum_pr_auc_delta",
            "minimum_precision_delta",
            "minimum_recall_delta",
            "maximum_expected_cost_per_transaction_increase",
            "maximum_brier_score_increase",
            "maximum_calibration_error_increase",
            "maximum_alert_rate_increase",
        ),
        "quality_gates.comparison",
    )
    _validate_ranges(absolute, comparison)
    minimum_test_rows = _require_positive_int(gates, "minimum_test_rows")
    minimum_test_positives = _require_positive_int(gates, "minimum_test_positives")
    return RetrainingQualityPolicy(
        absolute=absolute,
        minimum_test_rows=minimum_test_rows,
        minimum_test_positives=minimum_test_positives,
        comparison=comparison,
        policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
    )


def evaluate_retraining_quality(
    *,
    schema_contract_passed: bool,
    duplicate_transaction_id_count: int,
    split_overlap_count: int,
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
    test_calibration_error: float,
    test_row_count: int,
    test_positive_count: int,
    incumbent_metrics: dict[str, float],
    policy: RetrainingQualityPolicy,
) -> dict[str, Any]:
    """Evaluate absolute and same-test-set non-regression gates for one candidate."""

    checks: list[dict[str, Any]] = []
    absolute = policy.absolute
    checks.extend(
        [
            {
                "name": "schema.feature_contract",
                "actual": bool(schema_contract_passed),
                "operator": "==",
                "threshold": True,
                "passed": bool(schema_contract_passed),
            },
            {
                "name": "leakage.duplicate_transaction_ids",
                "actual": int(duplicate_transaction_id_count),
                "operator": "==",
                "threshold": 0,
                "passed": duplicate_transaction_id_count == 0,
            },
            {
                "name": "leakage.split_overlap",
                "actual": int(split_overlap_count),
                "operator": "==",
                "threshold": 0,
                "passed": split_overlap_count == 0,
            },
        ]
    )
    _add_minimum_check(
        checks,
        "validation.pr_auc",
        validation_metrics["pr_auc"],
        absolute["minimum_validation_pr_auc"],
    )
    _add_minimum_check(
        checks,
        "validation.precision",
        validation_metrics["precision"],
        absolute["minimum_validation_precision"],
    )
    _add_minimum_check(
        checks,
        "validation.recall",
        validation_metrics["recall"],
        absolute["minimum_validation_recall"],
    )
    _add_minimum_check(
        checks,
        "test.rows",
        test_row_count,
        policy.minimum_test_rows,
    )
    _add_minimum_check(
        checks,
        "test.positives",
        test_positive_count,
        policy.minimum_test_positives,
    )
    for metric_name in ("pr_auc", "precision", "recall"):
        _add_minimum_check(
            checks,
            f"test.{metric_name}",
            test_metrics[metric_name],
            absolute[f"minimum_test_{metric_name}"],
        )
    for metric_name in (
        "expected_cost_per_transaction",
        "brier_score",
        "calibration_error",
        "alert_rate",
    ):
        policy_key = {
            "expected_cost_per_transaction": "maximum_test_expected_cost_per_transaction",
            "brier_score": "maximum_test_brier_score",
            "calibration_error": "maximum_test_calibration_error",
            "alert_rate": "maximum_test_alert_rate",
        }[metric_name]
        _add_maximum_check(
            checks,
            f"test.{metric_name}",
            test_calibration_error
            if metric_name == "calibration_error"
            else test_metrics[metric_name],
            absolute[policy_key],
        )
    deltas = {
        "pr_auc": test_metrics["pr_auc"] - incumbent_metrics["pr_auc"],
        "precision": test_metrics["precision"] - incumbent_metrics["precision"],
        "recall": test_metrics["recall"] - incumbent_metrics["recall"],
        "expected_cost_per_transaction_increase": test_metrics["expected_cost_per_transaction"]
        - incumbent_metrics["expected_cost_per_transaction"],
        "brier_score_increase": test_metrics["brier_score"] - incumbent_metrics["brier_score"],
        "calibration_error_increase": test_calibration_error
        - incumbent_metrics["calibration_error"],
        "alert_rate_increase": test_metrics["alert_rate"] - incumbent_metrics["alert_rate"],
    }
    delta_limits = {
        "pr_auc": ("minimum_pr_auc_delta", "minimum"),
        "precision": ("minimum_precision_delta", "minimum"),
        "recall": ("minimum_recall_delta", "minimum"),
        "expected_cost_per_transaction_increase": (
            "maximum_expected_cost_per_transaction_increase",
            "maximum",
        ),
        "brier_score_increase": ("maximum_brier_score_increase", "maximum"),
        "calibration_error_increase": (
            "maximum_calibration_error_increase",
            "maximum",
        ),
        "alert_rate_increase": ("maximum_alert_rate_increase", "maximum"),
    }
    for metric_name, (policy_key, direction) in delta_limits.items():
        check_name = f"comparison.{metric_name}"
        if direction == "minimum":
            _add_minimum_check(
                checks, check_name, deltas[metric_name], policy.comparison[policy_key]
            )
        else:
            _add_maximum_check(
                checks, check_name, deltas[metric_name], policy.comparison[policy_key]
            )
    failed = [check["name"] for check in checks if not check["passed"]]
    return {
        "status": "passed" if not failed else "rejected",
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
        "incumbent_metrics": incumbent_metrics,
        "candidate_validation_metrics": validation_metrics,
        "candidate_test_metrics": {**test_metrics, "calibration_error": test_calibration_error},
        "test_row_count": test_row_count,
        "test_positive_count": test_positive_count,
    }


def expected_calibration_error(bins: list[dict[str, Any]]) -> float:
    """Calculate weighted absolute calibration error from equal-width calibration bins."""

    total = sum(int(item["count"]) for item in bins)
    if total <= 0:
        raise RetrainingError("Calibration data must contain at least one row")
    value = sum(
        int(item["count"])
        / total
        * abs(float(item["mean_predicted_probability"]) - float(item["observed_positive_rate"]))
        for item in bins
        if int(item["count"]) > 0
    )
    if not isfinite(value):
        raise RetrainingError("Calibration error is not finite")
    return float(value)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _require_number_mapping(value: Any, keys: tuple[str, ...], label: str) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise RetrainingError(f"{label} must contain exactly: {sorted(keys)}")
    normalized: dict[str, float] = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, int | float) or not isfinite(float(item)):
            raise RetrainingError(f"{label}.{key} must be a finite number")
        normalized[key] = float(item)
    return normalized


def _validate_ranges(absolute: dict[str, float], comparison: dict[str, float]) -> None:
    ratio_keys = (
        "minimum_validation_pr_auc",
        "minimum_validation_precision",
        "minimum_validation_recall",
        "minimum_test_pr_auc",
        "minimum_test_precision",
        "minimum_test_recall",
        "maximum_test_brier_score",
        "maximum_test_calibration_error",
        "maximum_test_alert_rate",
    )
    if any(not 0 <= absolute[key] <= 1 for key in ratio_keys):
        raise RetrainingError("Retraining metric thresholds must be between 0 and 1")
    if absolute["maximum_test_expected_cost_per_transaction"] < 0:
        raise RetrainingError("maximum_test_expected_cost_per_transaction must be non-negative")
    for key in (
        "maximum_expected_cost_per_transaction_increase",
        "maximum_brier_score_increase",
        "maximum_calibration_error_increase",
        "maximum_alert_rate_increase",
    ):
        if comparison[key] < 0:
            raise RetrainingError(f"{key} must be non-negative")


def _require_positive_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RetrainingError(f"quality_gates.{key} must be a positive integer")
    return value


def _add_minimum_check(
    checks: list[dict[str, Any]], name: str, actual: float, minimum: float
) -> None:
    checks.append(
        {
            "name": name,
            "actual": float(actual),
            "operator": ">=",
            "threshold": float(minimum),
            "passed": float(actual) >= float(minimum),
        }
    )


def _add_maximum_check(
    checks: list[dict[str, Any]], name: str, actual: float, maximum: float
) -> None:
    checks.append(
        {
            "name": name,
            "actual": float(actual),
            "operator": "<=",
            "threshold": float(maximum),
            "passed": float(actual) <= float(maximum),
        }
    )
