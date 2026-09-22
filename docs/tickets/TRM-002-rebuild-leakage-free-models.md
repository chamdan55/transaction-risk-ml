# TRM-002 — Rebuild Leakage-Free Dataset and Models

**Type:** Model correctness
**Priority:** P0
**Sprint:** 3.5 / Sprint 2 revalidation
**Dependencies:** TRM-001
**Status:** Implemented in code — Spark retraining and artifact verification pending runtime validation

## Problem

Existing artifacts and near-perfect metrics were produced with post-transaction features. They
cannot be used as evidence for pre-transaction risk scoring.

## Scope

- Regenerate feature splits using the contract introduced by TRM-001.
- Retrain Logistic Regression, Random Forest, and XGBoost.
- Preserve validation-only candidate selection and one-time final test evaluation.
- Evaluate whether combined negative undersampling and class weighting is justified; choose and
  document one deliberate strategy per model.
- Compare old and new metrics without presenting the old result as a valid production baseline.
- Generate a new evaluation report and model artifacts.
- Add regression tests proving forbidden features are absent from fitted preprocessors/artifacts.
- Persist model-ready split projections separately from the full audit feature schema.

## Non-Scope

- API serving.
- Drift monitoring.
- Automatic production promotion.

## Acceptance Criteria

- Generated model-ready splits contain only `transaction_id`, target, and contract-approved
  features; model preprocessors contain contract-approved features only.
- All three models train reproducibly from configuration.
- Validation selection and frozen-threshold test evaluation are preserved.
- Report clearly identifies dataset/contract version and sampling/weighting strategy.
- Old leakage-affected artifacts cannot be confused with the new candidate.
- Relevant data, model, and integration tests pass.

## Suggested Validation

```powershell
python -m pipelines.run_pipeline
python -m pipelines.train_models
pytest tests/unit tests/integration/test_training_pipeline.py
ruff check .
ruff format --check .
```

The Spark-dependent commands must run in a supported Java/Spark runtime. A sandbox loopback
failure is an environment blocker, not evidence that the model assertions passed.

## Prompt for an AI Agent

> Implement TRM-002 after verifying TRM-001 is complete. Read the architecture, sprint plan, and
> both tickets. Regenerate the feature data and retrain all configured models using only the
> approved pre-transaction contract. Audit the interaction between negative sampling and class
> weighting and implement a documented, configuration-driven strategy rather than silently
> applying both. Preserve chronological split semantics and validation-only candidate selection.
> Add artifact-level leakage assertions and update reports/docs with honest post-fix metrics. Do not
> promote a model to production. Preserve unrelated changes and report validation commands and
> results, distinguishing failed tests from tests blocked by the runtime.
