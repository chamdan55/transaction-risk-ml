from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import pandas as pd

from app.core.config import Settings
from app.services.model_provider import ModelMetadata
from ml.contracts.features import MODEL_FEATURE_COLUMNS
from ml.monitoring.contracts import DELAYED_LABEL_SCHEMA_VERSION, DelayedLabel, PredictionEvent
from ml.monitoring.store import PredictionEventStore
from pipelines.run_monitoring import run_monitoring_job


class ReadyProvider:
    is_ready = True
    metadata = ModelMetadata(
        model_name="random_forest",
        model_version="monitor-test-v1",
        model_source="local",
        threshold=0.5,
        threshold_policy="business_cost",
        feature_contract_version="pre-transaction-v1",
    )

    def load(self) -> None:
        pass

    def predict_scores(self, frame: pd.DataFrame) -> list[float]:
        return [0.1 if index % 2 == 0 else 0.9 for index in range(len(frame))]


class FakeReportResult:
    def save_html(self, path: str) -> None:
        Path(path).write_text("<html>summary only</html>", encoding="utf-8")

    def save_json(self, path: str) -> None:
        Path(path).write_text('{"summary": true}', encoding="utf-8")


class FakeReport:
    def __init__(self, presets: list[object]) -> None:
        assert len(presets) == 2

    def run(self, *, current_data: pd.DataFrame, reference_data: pd.DataFrame):
        assert "risk_score" in current_data
        assert "risk_score" in reference_data
        assert "feedback_id" not in current_data
        return FakeReportResult()


def _features(index: int) -> dict[str, str | int | float]:
    return {
        name: ("PAYMENT" if name == "transaction_type" else float(index))
        for name in MODEL_FEATURE_COLUMNS
    }


def test_monitoring_job_writes_version_segmented_drift_and_feedback_summary(
    tmp_path, monkeypatch
) -> None:
    class FakeDataSummaryPreset:
        pass

    class FakeDataDriftPreset:
        def __init__(self, *, columns: list[str]) -> None:
            assert columns == [*MODEL_FEATURE_COLUMNS, "risk_score"]

    evidently_module = ModuleType("evidently")
    evidently_module.Report = FakeReport
    presets_module = ModuleType("evidently.presets")
    presets_module.DataSummaryPreset = FakeDataSummaryPreset
    presets_module.DataDriftPreset = FakeDataDriftPreset
    monkeypatch.setitem(sys.modules, "evidently", evidently_module)
    monkeypatch.setitem(sys.modules, "evidently.presets", presets_module)

    now = datetime.now(UTC)
    database = tmp_path / "events.sqlite3"
    store = PredictionEventStore(database)
    feedback_ids = [uuid4(), uuid4(), uuid4()]
    events = [
        PredictionEvent(
            feedback_id=feedback_ids[0],
            event_time=now - timedelta(hours=1),
            model_name="random_forest",
            model_version="monitor-test-v1",
            feature_contract_version="pre-transaction-v1",
            features=_features(0),
            risk_score=0.9,
            decision_threshold=0.5,
            decision="review",
            latency_seconds=0.02,
        ),
        PredictionEvent(
            feedback_id=feedback_ids[1],
            event_time=now - timedelta(hours=2),
            model_name="random_forest",
            model_version="monitor-test-v1",
            feature_contract_version="pre-transaction-v1",
            features=_features(1),
            risk_score=0.1,
            decision_threshold=0.5,
            decision="allow",
            latency_seconds=0.03,
        ),
        PredictionEvent(
            feedback_id=feedback_ids[2],
            event_time=now - timedelta(days=10),
            model_name="random_forest",
            model_version="monitor-test-v1",
            feature_contract_version="pre-transaction-v1",
            features=_features(2),
            risk_score=0.2,
            decision_threshold=0.5,
            decision="allow",
            latency_seconds=0.04,
        ),
    ]
    store._persist_events_sync(events)
    store._record_label_sync(
        DelayedLabel(
            schema_version=DELAYED_LABEL_SCHEMA_VERSION,
            feedback_id=feedback_ids[0],
            label=1,
            label_time=now - timedelta(minutes=5),
        )
    )
    store._record_label_sync(
        DelayedLabel(
            schema_version=DELAYED_LABEL_SCHEMA_VERSION,
            feedback_id=feedback_ids[1],
            label=0,
            label_time=now - timedelta(minutes=4),
        )
    )
    store._record_label_sync(
        DelayedLabel(
            schema_version=DELAYED_LABEL_SCHEMA_VERSION,
            feedback_id=uuid4(),
            label=1,
            label_time=now - timedelta(minutes=3),
        )
    )

    reference_path = tmp_path / "validation.parquet"
    pd.DataFrame([_features(index) for index in range(10)]).to_parquet(reference_path)
    output = tmp_path / "reports"
    summary = run_monitoring_job(
        settings=Settings(
            ml_monitoring_database_path=database,
            ml_monitoring_retention_days=90,
        ),
        reference_data=reference_path,
        output_directory=output,
        window_days=30,
        label_grace_days=7,
        max_report_rows=20,
        max_events=100,
        provider=ReadyProvider(),
        now=now,
    )

    assert summary["events"]["joined_labels"] == 2
    assert summary["events"]["unmatched_or_invalid_labels"] == 1
    assert summary["events"]["late_unlabeled_predictions"] == 1
    assert summary["segments"][0]["drift_status"] == "generated"
    assert summary["segments"][0]["performance"]["status"] == "computed"
    assert (output / "latest-summary.json").is_file()
