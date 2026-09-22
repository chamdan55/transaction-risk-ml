# TRM-013 — Implement Controlled Retraining, Promotion, and Rollback

**Type:** MLOps lifecycle capability
**Priority:** P1
**Sprint:** 7
**Dependencies:** TRM-006, TRM-012
**Status:** Blocked

## Problem

The lifecycle is incomplete without reproducible retraining, but drift alone must not automatically
replace the production model.

## Scope

- Add a retraining entry point consuming an immutable approved dataset manifest.
- Define gates for leakage/schema checks, PR-AUC, recall, precision, cost, calibration, and alert rate.
- Compare a candidate against configured absolute thresholds and the current production model.
- Register passing models as candidate; keep rejected runs/artifacts traceable.
- Reuse explicit approver/reason metadata for staging/production aliases.
- Implement controlled API rollout and rollback to the previous image/model alias.
- Add normal, drift, passing candidate, rejected candidate, and rollback demo scenarios.
- Schedule retraining with GitHub Actions or Kubernetes CronJob; keep promotion human-gated.

## Non-Scope

- Fully autonomous production promotion.
- Airflow/Kubeflow unless the project scope materially changes.

## Acceptance Criteria

- Same manifest/config/code inputs produce traceable retraining runs.
- Failed quality gates cannot move production/staging aliases.
- Passing candidate still requires explicit approval.
- Promotion records reviewer, reason, prior version, new version, and time.
- Rollback restores a previously known-good model and service readiness.
- End-to-end demo covers both reject and promote paths.

## Suggested Validation

```powershell
pytest tests/integration/test_retraining_pipeline.py
pytest tests/integration/test_promotion_gates.py tests/integration/test_rollback.py
python -m pipelines.retrain_model
```

## Prompt for an AI Agent

> Implement TRM-013 only after lineage/evaluation and ML monitoring are complete. Build a
> reproducible retraining entry point consuming an immutable dataset manifest. Enforce leakage,
> schema, performance, cost, calibration, and operational-volume gates; compare against both
> absolute constraints and the active production version. Register passing output only as a
> candidate and preserve rejected runs for audit. Require explicit approver and reason for promotion,
> then implement and test controlled rollout and rollback. Use a scheduled GitHub Actions job or
> Kubernetes CronJob rather than adding Airflow. Demonstrate reject, promote, and rollback paths and
> report all evidence without claiming autonomous production safety.
