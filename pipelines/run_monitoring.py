"""Produce version-segmented batch quality and delayed-performance reports."""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import Settings
from app.services.model_provider import ModelProvider
from ml.contracts.features import MODEL_FEATURE_COLUMNS, validate_model_feature_columns
from ml.evaluation.metrics import (
    EvaluationError,
    build_calibration_data,
    evaluate_binary_predictions,
)
from ml.monitoring.contracts import PREDICTION_EVENT_SCHEMA_VERSION
from ml.monitoring.store import PredictionEventStore

LOGGER = logging.getLogger(__name__)
DEFAULT_REFERENCE_DATA = Path("data/processed/features/validation")
DEFAULT_OUTPUT_DIRECTORY = Path("artifacts/monitoring/reports")
DEFAULT_MAX_REPORT_ROWS = 20_000
DEFAULT_MAX_EVENTS = 100_000


def run_monitoring_job(
    *,
    settings: Settings | None = None,
    reference_data: str | Path = DEFAULT_REFERENCE_DATA,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    window_days: int = 7,
    label_grace_days: int = 7,
    max_report_rows: int = DEFAULT_MAX_REPORT_ROWS,
    max_events: int = DEFAULT_MAX_EVENTS,
    provider: Any | None = None,
    report_factory: Any | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Read a bounded snapshot, segment it, and write Evidently plus metric artifacts.

    This function is intended to run as a separate scheduled process. It never runs on the
    prediction request path and it never changes model aliases or retrains a model.
    """

    if window_days < 1 or label_grace_days < 1 or max_report_rows < 2 or max_events < 1:
        raise ValueError("window and sample limits must be positive; report sample must be >= 2")

    resolved_settings = settings or Settings()
    clock = (now or datetime.now(UTC)).astimezone(UTC)
    report_directory = Path(output_directory)
    report_directory.mkdir(parents=True, exist_ok=True)

    resolved_provider = provider or ModelProvider(resolved_settings)
    if not resolved_provider.is_ready:
        resolved_provider.load()
    if not resolved_provider.is_ready:
        raise RuntimeError("Configured model is not ready for monitoring report generation")
    metadata = resolved_provider.metadata

    store = PredictionEventStore(
        resolved_settings.ml_monitoring_database_path,
        queue_maxsize=resolved_settings.ml_monitoring_queue_maxsize,
        retention_days=resolved_settings.ml_monitoring_retention_days,
    )
    snapshot = store.read_snapshot(
        window_days=window_days,
        retention_days=resolved_settings.ml_monitoring_retention_days,
        label_grace_days=label_grace_days,
        max_events=max_events,
    )
    segments = _segment_events(snapshot["events"])
    reference_path = Path(reference_data)
    reference: pd.DataFrame | None = None
    active_segment_present = any(
        key
        == (
            metadata.model_name,
            metadata.model_version,
            metadata.feature_contract_version,
        )
        and len(rows) >= 2
        for key, rows in segments.items()
    )
    if active_segment_present:
        feature_columns = validate_model_feature_columns(MODEL_FEATURE_COLUMNS)
        reference_frame = pd.read_parquet(reference_path, columns=list(feature_columns))
        missing = sorted(set(feature_columns).difference(reference_frame.columns))
        if missing:
            raise ValueError(f"Reference split is missing contract features: {missing}")
        if len(reference_frame) < 2:
            raise ValueError("Reference split must contain at least two rows")
        reference_frame = _sample_frame(reference_frame, max_report_rows, seed=42)
        reference_features = reference_frame.loc[:, list(feature_columns)].copy()
        reference_scores = resolved_provider.predict_scores(reference_features)
        reference = reference_features.copy()
        reference["risk_score"] = reference_scores

    segment_reports = []
    for key, rows in sorted(segments.items()):
        segment = _build_segment_summary(
            key,
            rows,
            metadata=metadata,
            reference=reference,
            report_directory=report_directory,
            max_report_rows=max_report_rows,
            report_factory=report_factory,
        )
        segment_reports.append(segment)

    summary = {
        "report_schema_version": "ml-monitoring-summary-v1",
        "generated_at": clock.isoformat(),
        "window_start": snapshot["window_start"],
        "retention_start": snapshot["retention_start"],
        "retention_days": resolved_settings.ml_monitoring_retention_days,
        "label_grace_days": label_grace_days,
        "reference_split": reference_path.as_posix(),
        "reference_model": {
            "model_name": metadata.model_name,
            "model_version": metadata.model_version,
            "feature_contract_version": metadata.feature_contract_version,
            "threshold": metadata.threshold,
        },
        "events": {
            "schema_version": PREDICTION_EVENT_SCHEMA_VERSION,
            "retained_in_window": snapshot["event_count"],
            "sampled_for_reports": sum(
                item.get("current_sample_count", 0) for item in segment_reports
            ),
            "joined_labels": snapshot["joined_label_count"],
            "unmatched_or_invalid_labels": snapshot["unmatched_label_count"],
            "late_unlabeled_predictions": snapshot["late_unlabeled_count"],
            "report_input_truncated": snapshot["truncated"],
        },
        "segments": segment_reports,
    }
    _write_json(report_directory / "latest-summary.json", summary)
    LOGGER.info(
        "ML monitoring report completed: events=%s segments=%s output=%s",
        snapshot["event_count"],
        len(segment_reports),
        report_directory,
    )
    return summary


def _segment_events(
    events: list[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    segments: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for event in events:
        key = (
            event["model_name"],
            event["model_version"],
            event["feature_contract_version"],
        )
        segments.setdefault(key, []).append(event)
    return segments


def _build_segment_summary(
    key: tuple[str, str, str],
    rows: list[dict[str, Any]],
    *,
    metadata: Any,
    reference: pd.DataFrame | None,
    report_directory: Path,
    max_report_rows: int,
    report_factory: Any | None,
) -> dict[str, Any]:
    model_name, model_version, contract_version = key
    segment = {
        "model_name": model_name,
        "model_version": model_version,
        "feature_contract_version": contract_version,
        "event_count": len(rows),
        "labeled_count": sum(row["label"] is not None for row in rows),
    }
    compatible_model = (
        model_name == metadata.model_name
        and model_version == metadata.model_version
        and contract_version == metadata.feature_contract_version
    )
    if not compatible_model:
        segment["drift_status"] = "skipped_reference_model_mismatch"
    elif len(rows) < 2:
        segment["drift_status"] = "insufficient_current_rows"
    elif reference is None:
        segment["drift_status"] = "reference_unavailable"
    else:
        stage = "prepare_data"
        try:
            current_rows = _sample_records(rows, max_report_rows)
            current_features = pd.DataFrame([row["features"] for row in current_rows])
            current = current_features.loc[:, list(MODEL_FEATURE_COLUMNS)].copy()
            current["risk_score"] = [float(row["risk_score"]) for row in current_rows]
            segment["current_sample_count"] = len(current)
            stem = "-".join(_slug(value) for value in key)
            html_path = report_directory / f"drift-{stem}.html"
            json_path = report_directory / f"drift-{stem}.json"
            stage = "build_report"
            if report_factory is None:
                from evidently import Report
                from evidently.presets import DataDriftPreset, DataSummaryPreset

                report = Report(
                    [
                        DataSummaryPreset(),
                        DataDriftPreset(columns=[*MODEL_FEATURE_COLUMNS, "risk_score"]),
                    ]
                )
            else:
                report = report_factory()
            stage = "run_report"
            result = report.run(current_data=current, reference_data=reference)
            stage = "save_report"
            result.save_html(str(html_path))
            result.save_json(str(json_path))
            segment["drift_status"] = "generated"
            segment["drift_report_html"] = html_path.name
            segment["drift_report_json"] = json_path.name
        except Exception as exc:
            LOGGER.error(
                "Evidently report failed for model segment=%s stage=%s error_type=%s",
                key,
                stage,
                type(exc).__name__,
            )
            segment["drift_status"] = "failed"
            segment["drift_error_stage"] = stage
            segment["drift_error"] = type(exc).__name__

    labeled_rows = [row for row in rows if row["label"] is not None]
    if not labeled_rows:
        segment["performance"] = {"status": "awaiting_labels", "count": 0}
    else:
        labels = [int(row["label"]) for row in labeled_rows]
        scores = [float(row["risk_score"]) for row in labeled_rows]
        threshold = float(labeled_rows[0]["decision_threshold"])
        performance: dict[str, Any] = {
            "status": "computed",
            "count": len(labeled_rows),
            "decision_threshold": threshold,
            "calibration_bins": build_calibration_data(labels, scores),
        }
        try:
            performance["classification_metrics"] = evaluate_binary_predictions(
                labels, scores, threshold=threshold
            ).as_dict()
        except EvaluationError:
            performance["status"] = "awaiting_both_label_classes"
        segment["performance"] = performance
    return segment


def _sample_frame(frame: pd.DataFrame, max_rows: int, *, seed: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.copy()
    return frame.sample(n=max_rows, random_state=seed).sort_index().reset_index(drop=True)


def _sample_records(rows: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    if len(rows) <= max_rows:
        return rows
    # Stable striding avoids exposing identifiers to a random-number generator or output artifact.
    stride = len(rows) / max_rows
    return [rows[int(index * stride)] for index in range(max_rows)]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._")
    return slug[:80] or "unknown"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate asynchronous ML observability reports")
    parser.add_argument("--reference-data", type=Path, default=DEFAULT_REFERENCE_DATA)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--label-grace-days", type=int, default=7)
    parser.add_argument("--max-report-rows", type=int, default=DEFAULT_MAX_REPORT_ROWS)
    parser.add_argument("--max-events", type=int, default=DEFAULT_MAX_EVENTS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings()
    summary = run_monitoring_job(
        settings=settings,
        reference_data=args.reference_data,
        output_directory=args.output_directory,
        window_days=args.window_days,
        label_grace_days=args.label_grace_days,
        max_report_rows=args.max_report_rows,
        max_events=args.max_events,
    )
    if any(segment.get("drift_status") == "failed" for segment in summary["segments"]):
        raise SystemExit("At least one Evidently segment failed; inspect the report summary/logs")


if __name__ == "__main__":
    main()
