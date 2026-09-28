# TRM-013 — Implement Controlled Retraining, Promotion, and Rollback

**Type:** MLOps lifecycle capability
**Priority:** P1
**Sprint:** 7
**Dependencies:** TRM-006, TRM-012, TRM-016
**Status:** Implemented — owner Pytest and containerized runtime scenarios pending

## Problem

The lifecycle is incomplete without reproducible retraining, but drift alone must not automatically
replace the production model. TRM-016 first establishes a shared containerized tracking/registry
endpoint for training and promotion; TRM-013 must use that same endpoint.

## Scope

- Add a create-once approved-manifest generator and a retraining entry point that verifies data,
  schema, config, and training-code fingerprints before fitting.
- Gate schema/leakage checks, validation and test PR-AUC/precision/recall, expected business cost
  per evaluated transaction, Brier score, expected calibration error, alert rate, sample size, and
  deltas against the production model scored on the same held-out split.
- Register and move only passing retrains to the `candidate` alias; preserve rejected runs, reports,
  and unregistered model artifacts in MLflow.
- Record approver, reason, prior/new alias version, and UTC timestamp for staging/production changes.
- Export the explicitly approved production alias into a versioned local serving bundle; perform a
  readiness-gated kind rollout and restore the previous Deployment revision and MLflow alias on
  rollout failure.
- Add normal, drift-review, passing candidate, rejected candidate, and rollback procedures.
- Add a suspended-by-default Kubernetes CronJob consuming a mounted, separately approved manifest;
  keep production promotion human-gated.

## Non-Scope

- Fully autonomous production promotion.
- Airflow/Kubeflow unless the project scope materially changes.

## Acceptance Criteria

- Same manifest/config/code inputs produce traceable retraining runs.
- Failed schema/leakage or model-quality gates are traceable and cannot create a model version or move
  candidate/staging/production aliases.
- Passing candidate still requires explicit approval.
- Promotion records reviewer, reason, prior version, new version, and UTC time per alias.
- Rollback restores a previously known-good image/model alias and waits for service readiness.
- End-to-end owner validation covers reject, promote, rollout, and rollback paths.
- The scheduled job remains disabled until its approved manifest, data PVC, secured reachable MLflow
  endpoint, and resource profile are provisioned.

## Suggested Validation

```powershell
# Owner runs focused tests; implementing agent must not run pytest.
pytest tests/unit/test_retraining.py tests/unit/test_promotion.py tests/unit/test_rollout_model.py
pytest tests/integration/test_mlflow_training_run.py

make up-tracking
make tracking-smoke
make prepare-retraining-manifest APPROVED_BY="risk-reviewer" APPROVAL_REASON="Reviewed snapshot" `
  APPROVED_MANIFEST="artifacts/retraining/approved-manifest-demo.json"
make retrain-tracking APPROVED_MANIFEST="artifacts/retraining/approved-manifest-demo.json"
# Inspect the MLflow gate report. Promote only after review, then export and roll out to kind.
make promote-tracking MODEL_VERSION="<reviewed-version>" PROMOTION_STAGE=production `
  APPROVED_BY="risk-reviewer" PROMOTION_REASON="Reviewed gate report"
make export-production-model
make kind-rollout MODEL_BUNDLE_PATH="<exported-bundle-directory>"
```

The committed policy thresholds are provisional PaySim demo limits and require risk-owner review
before real production decisions. `make retrain-tracking` requires an existing validated production
alias. Rollout rollback requires a prior production version. The Kubernetes CronJob is suspended by
default; see `docs/retraining.md` for its PVC, ConfigMap, Secret, and reachable tracking prerequisites.

## Prompt for an AI Agent

> Implement TRM-013 only after lineage/evaluation, ML monitoring, and TRM-016 containerized tracking
> integration are available. Build a create-once approved dataset manifest that binds split content,
> schema, model config, and source fingerprint, then fail closed if any input changes. Compare the
> challenger and active production model on the same held-out split against explicit PR-AUC,
> precision, recall, business-cost, calibration, alert-rate, schema/leakage, and sample-size gates.
> Keep rejected runs/artifacts traceable without creating versions or moving registry aliases.
> Require named approval and reason for staging/production, record prior/new versions and timestamps,
> and export approved versions into content-tagged serving bundles. Roll out with readiness checks
> and restore both the previous Kubernetes revision and MLflow alias if rollout fails. Provide a
> suspended-by-default CronJob that consumes an externally approved manifest; do not create data
> approvals, promote production, or claim that demo thresholds are business-approved. The owner runs
> Pytest and the containerized end-to-end scenarios; report them as pending until evidence arrives.
