# TRM-011 — Add Runtime Metrics, Dashboards, and Alerts

**Type:** Observability capability
**Priority:** P1
**Sprint:** 6
**Dependencies:** TRM-008, TRM-009
**Status:** Blocked

## Problem

Basic text logging is insufficient to identify latency degradation, error spikes, readiness loss,
or model-loading failures.

## Scope

- Instrument request count, duration histogram, in-flight requests, response status, validation
  rejection, model load failures, readiness, and prediction decision counts.
- Expose Prometheus-compatible `/metrics` with safe, bounded labels.
- Add Prometheus configuration, Grafana dashboard provisioning, and actionable alert rules.
- Add structured JSON log option and correlation between request logs and errors.
- Document metric names, units, label policy, and alert runbooks.
- Test that metrics contain no transaction/account IDs or other high-cardinality labels.

## Non-Scope

- Feature or prediction drift; handled by TRM-012.
- Distributed tracing unless a second meaningful online service is introduced.

## Acceptance Criteria

- Dashboard displays request rate, p95 latency, error rate, readiness, and active model version.
- Alerts cover sustained errors, latency, and no-ready-instance/model-load failure scenarios.
- Metrics tests prove bounded label cardinality and privacy policy.
- Monitoring outages do not break prediction requests.

## Suggested Validation

```powershell
pytest tests/unit/test_metrics.py tests/integration/test_observability.py
docker compose --profile monitoring up
```

## Prompt for an AI Agent

> Implement TRM-011 after the secured containerized API exists. Add Prometheus metrics with bounded,
> non-sensitive labels, provision a concise Grafana dashboard, and create actionable alert rules and
> runbooks. Include model-load/readiness and validation failures, not only HTTP counters. Ensure
> telemetry failures cannot fail the prediction path, and test that account/transaction IDs never
> become labels. Avoid distributed tracing unless the architecture now has multiple online hops.
> Validate the monitoring Compose profile and report dashboard/alert evidence.
