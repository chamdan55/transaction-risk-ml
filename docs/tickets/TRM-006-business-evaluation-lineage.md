# TRM-006 — Add Business Evaluation, Calibration, and Dataset Lineage

**Type:** Model governance enhancement
**Priority:** P1
**Sprint:** 3.5 / Sprint 2–3 hardening
**Dependencies:** TRM-001, TRM-002
**Status:** Implemented in code — owner runtime validation pending with TRM-014 correction

## Problem

Current threshold analysis defaults false-positive and false-negative cost to the same value, selects
threshold by F1, lacks probability-calibration reporting, and identifies datasets mostly by name and
row count.

## Scope

- Add validated configuration for false-positive cost, false-negative cost, minimum recall,
  threshold selection strategy, and final-test sampling policy.
- Report PR-AUC, ROC-AUC, precision, recall, F1, confusion matrix, Brier score, calibration data,
  alert rate, and expected cost.
- Use complete final test data by default.
- Add dataset/feature manifests with content/schema/config/code fingerprints and time ranges.
- Attach manifests, feature-contract version, model signature, input example, and evaluation artifacts
  to MLflow run/model metadata.
- Document assumptions and limitations in a generated or versioned model card.

Implementation notes:

- Default business costs remain explicit assumptions (`false_positive=1.0`, `false_negative=5.0`)
  until a business owner supplies approved values.
- Final-test sampling defaults to `full`; `exact`/`hash` are opt-in and are recorded in the report.
- The model card deliberately does not promote a candidate automatically.

## Non-Scope

- Choosing real monetary cost values without business input; defaults must be labeled assumptions.
- Automatic production promotion.

## Acceptance Criteria

- Threshold selection follows configured strategy and constraints.
- Calibration is measured before scores are described as probabilities.
- Final test evaluation is full-period by default and cannot influence model selection.
- Dataset identity changes when source content, schema, feature contract, or config changes.
- Registry model version links to manifest and evaluation artifacts.
- Tests cover configuration errors, lineage stability, and selection behavior.

## Suggested Validation

```powershell
pytest tests/unit/test_evaluation_metrics.py tests/unit/test_threshold.py tests/unit/test_selection.py tests/unit/test_lineage.py
pytest tests/unit/test_tracking_metadata.py tests/integration/test_training_pipeline.py
pytest tests/integration/test_mlflow_training_run.py
```

## Prompt for an AI Agent

> Implement TRM-006 after the leakage-free dataset/model work is complete. Make business costs,
> recall constraints, threshold strategy, and final-test sampling explicit validated configuration.
> Add calibration and alert-volume reporting without using test data for tuning. Create stable
> dataset and feature manifests that fingerprint content/schema/config/code/time boundaries, and
> propagate their identifiers plus model signature and input example into MLflow artifacts and
> version metadata. Document cost values as assumptions unless supplied by a business owner. Add
> focused tests and a concise model card. Never auto-promote the candidate, and report exact
> validation results.
