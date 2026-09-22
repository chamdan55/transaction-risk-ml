# TRM-012 — Add Prediction Events, Label Feedback, and Drift Jobs

**Type:** ML observability capability
**Priority:** P1
**Sprint:** 6
**Dependencies:** TRM-007, TRM-011
**Status:** Blocked

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
python -m pipelines.run_monitoring
```

## Prompt for an AI Agent

> Implement TRM-012 after serving and runtime observability are stable. Design a versioned,
> privacy-aware prediction event and delayed-label contract with pseudonymous IDs and explicit
> retention. Persist events off the synchronous critical path using the simplest reliable mechanism
> suitable for the local profile; do not introduce Kafka without measured need. Implement
> idempotent label joins and scheduled Evidently data-quality, drift, calibration, and performance
> jobs segmented by model/contract version. Add contract/integration tests and failure-isolation
> tests. Do not trigger retraining in this ticket. Report artifacts and validation evidence.
