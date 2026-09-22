# TRM-014 — Isolate Per-Model Quality Gates and Preserve Rejected Evaluations

**Type:** Correctness and model-governance defect
**Priority:** P0
**Sprint:** 3.5 / Sprint 2–3 hardening
**Dependencies:** TRM-006
**Status:** Completed — owner Pytest and full-training validation passed

## Problem

A full-data training run proved that candidate selection fails as soon as the first model has no
threshold satisfying `minimum_recall`, even when later models pass the same gate. In the observed
run, Logistic Regression reached a maximum candidate recall of `0.768953` and correctly failed the
`0.80` constraint, while Random Forest and XGBoost reached recall `1.0` at every configured
threshold. The pipeline nevertheless stopped before selecting either eligible model.

The defect is repeated in three places:

- `select_production_candidate()` treats one model's threshold rejection as a pipeline-wide error.
- `_build_validation_report()` selects a threshold again for every model and would fail on the same
  rejected model after candidate selection is fixed.
- `_track_training_run()` also selects a threshold again and currently tags every validation run as
  `candidate`, including models that fail the configured quality gate.

This is not evidence that all trained models are inadequate. It is a failure to isolate a
per-model rejection from the model-set selection decision.

## Decision

- A quality constraint is evaluated independently for each model.
- A model with no eligible threshold is retained as a traceable `rejected` evaluation and excluded
  from production-candidate ranking.
- The pipeline fails only when no model remains eligible.
- Eligible models continue to be ranked by the configured `primary_metric`; the configured
  threshold strategy chooses the threshold within each model. This ticket does not silently change
  model ranking from PR-AUC to expected cost.
- Invalid input, unsupported metrics/strategies, malformed probabilities, and invalid cost
  configuration remain hard failures. Only the expected "no threshold satisfies the quality
  constraint" outcome is treated as a model rejection.

## Scope

- Introduce one reusable validation-result representation containing the model name, complete
  threshold sweep, eligibility status, rejection reason, maximum candidate recall, and optional
  selected threshold.
- Compute each model's threshold analysis once and reuse it for candidate selection, console logs,
  the JSON evaluation report, and MLflow logging.
- Exclude rejected models from candidate ranking without hiding their ranking metrics or threshold
  results.
- Preserve the existing primary-metric and expected-cost ordering, with an explicit stable model-name
  fallback so exact ties do not depend on mapping insertion order.
- Raise an aggregate, actionable `EvaluationError` only when all models are rejected. Include the
  configured minimum recall and each model's maximum candidate recall in the error.
- Preserve validation-only selection and evaluate the frozen selected threshold on the final test
  set exactly once.
- Record `candidate_status=rejected` plus the rejection reason for rejected MLflow validation runs;
  only the selected model may be registered as the production candidate.
- Keep the existing console threshold table and make its PASS/FAIL status derive from the same
  reusable validation result used by selection.
- Add focused unit and integration tests for mixed, all-passing, and all-rejected model sets.

## Non-Scope

- Lowering `minimum_recall`, adding new threshold candidates, or changing business-cost assumptions.
- Changing model ranking from configured `primary_metric` to expected cost without an explicit
  product/risk decision.
- Declaring the near-perfect tree-model validation metrics production-valid.
- Temporal backtesting, probability recalibration, or remediation of the observed class-prevalence
  shift between train, validation, and test; those require final-test evidence after this blocker is
  removed.
- Automatic staging or production promotion.

## Acceptance Criteria

- Given one rejected model and at least one eligible model, training selects from eligible models
  and does not raise because of the rejected model.
- Given the observed ordering (rejected Logistic Regression before eligible Random Forest and
  XGBoost), selection completes deterministically and is independent of dictionary insertion order.
- Given all models rejected, selection raises one aggregate error containing every model name,
  maximum candidate recall, and the configured minimum recall.
- Invalid strategies, unsupported metrics, bad probabilities, and invalid costs still fail fast and
  are not mislabeled as quality-gate rejection.
- The evaluation report includes every model, its full threshold sweep, eligibility status,
  rejection reason when applicable, and selected threshold only when one exists.
- MLflow records rejected model runs as `candidate_status=rejected`; they are never registered.
- The selected candidate still uses validation only, and final test data does not affect model or
  threshold selection.
- Tests cover mixed eligibility, all rejected, order independence, report serialization, and MLflow
  rejected-run metadata.

## Suggested Validation

The repository owner runs Pytest. The implementing agent should add the tests but restrict its own
validation to static checks unless the owner changes that instruction.

```powershell
# Owner-executed tests
pytest tests/unit/test_selection.py tests/unit/test_tracking_logging.py
pytest tests/integration/test_training_pipeline.py

# Agent-executed static checks
ruff check .
ruff format --check .
python -m compileall -q app ml pipelines tests
git diff --check
```

## Prompt for an AI Agent

> Implement TRM-014 in the transaction-risk-ml repository. First read
> `docs/arsitektur_dan_tech_stack.md`, `docs/sprint_docs.md`, TRM-006, and this ticket, then inspect
> every caller of `analyze_thresholds()` and `select_threshold()`. Refactor validation so each
> model's threshold sweep, eligibility, optional selected threshold, and rejection reason are
> computed once and reused by candidate selection, logs, JSON reporting, and MLflow. A model that
> cannot meet `minimum_recall` must be recorded as rejected and excluded from ranking; it must not
> abort the pipeline while another model is eligible. Fail with an aggregate diagnostic only when
> all models are rejected. Do not broadly catch `EvaluationError`: invalid configuration, metrics,
> costs, targets, and probabilities remain hard failures. Preserve PR-AUC as the configured
> cross-model ranking metric, business-cost threshold selection within each model, validation-only
> tuning, frozen-threshold final-test evaluation, and explicit promotion. Add tests for mixed
> eligibility, all-rejected diagnostics, order independence, report contents, and rejected MLflow
> metadata. The repository owner has reserved Pytest execution; do not run Pytest. Run Ruff, format
> check, compileall, and `git diff --check`, then report exact evidence and remaining runtime
> validation.
