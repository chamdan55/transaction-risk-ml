# TRM-012 — Add Prediction Events, Label Feedback, and Drift Jobs

**Type:** ML observability capability
**Priority:** P1
**Sprint:** 6
**Dependencies:** TRM-007, TRM-011
**Status:** Completed — owner pytest and Podman monitoring report validated

## Problem

ML drift and delayed performance cannot be measured without a versioned, privacy-aware prediction
event and a reliable way to join later ground-truth labels.

## Scope

- Define a versioned prediction event containing pseudonymous ID, event time, contract/model version,
  approved monitoring features, score, decision, and latency.
- Define delayed-label ingestion and deterministic join semantics.
- Persist events without blocking synchronous inference; document loss/retry behavior.
- Add retention/redaction rules.
- Build scheduled Evidently jobs for data quality, feature drift, prediction drift, calibration, and
  delayed performance when labels exist.
- Segment reports by model and feature-contract version.
- Export report summaries as bounded metrics or artifacts for Grafana/inspection.

## Non-Scope

- Kafka unless measured event volume proves it necessary.
- Automatic retraining or promotion.

## Acceptance Criteria

- Prediction and label schemas are versioned and contract-tested.
- Raw account IDs and unrestricted payloads are not retained.
- Label joins are idempotent and report unmatched/late records.
- Evidently job is reproducible and asynchronous.
- Monitoring/report failure does not affect API readiness or prediction.

## Suggested Validation

```powershell
pytest tests/contract/test_prediction_events.py
pytest tests/integration/test_feedback_pipeline.py tests/integration/test_drift_job.py
python -m pip install -e ".[monitoring]" -c requirements/constraints-py312.txt
python -m pipelines.run_monitoring --reference-data data/processed/features/validation
# If API is running in Podman Compose, run the report in the shared event volume:
make monitoring-report-compose
```

## Implementation Note

Successful API predictions now return a random `feedback_id` and enqueue an exact, versioned
`prediction-event-v1` feature snapshot to a bounded background writer. The writer persists into SQLite
with retries, deduplication, retention, and a bounded shutdown drain; queue overflow and process-crash
loss are documented and instrumented. `POST /v1/feedback/labels` accepts strict `delayed-label-v1`
records, permits delivery before the event writer flushes, accepts same-label retries, and rejects a
conflicting final label. Neither contract accepts account IDs or unrestricted payloads.

`pipelines.run_monitoring` is a separate scheduled process. It compares sampled recent features and
prediction scores with the validation split using Evidently, segments artifacts by model and contract,
and computes calibration/performance when delayed labels exist. The JSON summary includes unmatched labels
and predictions past the label grace period. Local output is written under `artifacts/monitoring/reports/`;
no retraining or model promotion is triggered. The container profile uses the same persistent SQLite
volume as the API; production multi-replica durability remains outside this local SQLite design.

Implementation and contract/integration tests are in place, including a concurrency case for idempotent
same-label retries and a regression check for Evidently's `DataDriftPreset(columns=...)` API. On
2026-09-24, the report job completed against the existing Compose event volume with two current events
and two joined labels, producing HTML, JSON, and `latest-summary.json` with `drift_status=generated`.
The first run exposed an Evidently 0.7.21 API mismatch: `DataDriftPreset` accepts `columns`, not
`column`. After the fix, the monitoring image rebuilt and the standard Compose report job completed
with exit code 0. The owner reported a green pytest run and reran `make monitoring-report-compose`
successfully on 2026-09-24; its report recorded two events, two joined labels, and
`drift_status=generated`. This validates the pipeline, while drift and performance values based on two
demo predictions are not statistically meaningful. Configure a recurring scheduler and collect
representative, genuinely labeled data when operating beyond the local demo. The report requires at
least two current events for an active model/contract segment before drift comparison is generated.

## Prompt for an AI Agent

> Implement TRM-012 after serving and runtime observability are stable. Design a versioned,
> privacy-aware prediction event and delayed-label contract with pseudonymous IDs and explicit
> retention. Persist events off the synchronous critical path using the simplest reliable mechanism
> suitable for the local profile; do not introduce Kafka without measured need. Implement
> idempotent label joins and scheduled Evidently data-quality, drift, calibration, and performance
> jobs segmented by model/contract version. Add contract/integration tests and failure-isolation
> tests. Do not trigger retraining in this ticket. Report artifacts and validation evidence.
