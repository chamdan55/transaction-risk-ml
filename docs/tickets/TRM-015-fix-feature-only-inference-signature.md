# TRM-015 — Fix Feature-Only Inference and Enforce MLflow Signature Validation

**Type:** Correctness, serving-contract, and registry-safety defect
**Priority:** P0
**Sprint:** 3.5 / Sprint 2–3 hardening
**Dependencies:** TRM-002, TRM-003, TRM-006, TRM-014
**Status:** Implemented in code — owner Pytest and full-training validation pending

## Problem

The full training pipeline now completes candidate selection and final-test evaluation, but MLflow
signature inference and serving-input validation fail with:

```text
FeaturePreparationError: Feature frame is missing required columns: ['is_fraud']
```

The training pipeline correctly builds a feature-only `input_example`; an inference request must
never contain the target. However, `FeaturePreprocessor.transform()` delegates to
`_select_features()`, which currently requires both model features and the training target. The
model wrappers therefore cannot run `predict()` or `predict_proba()` on a real serving payload.

The pipeline catches signature inference failure, logs only a warning, and continues to register
the model and update the `candidate` alias. The observed run created registered model version `2`
despite its invalid serving example and missing validated signature. This is a registry-safety bug,
not a model-training failure and not related to the intentional MLflow 3.x `name` parameter.

## Decision

- Training requires features plus target; inference requires exactly the ordered model features and
  must not require or accept the target as a model input.
- Feature selection and target extraction are separate validation concerns.
- A missing/invalid MLflow signature or invalid serving input example is a hard candidate-publication
  failure.
- The `candidate` alias may move only after the logged model is loadable and successfully predicts
  from the feature-only serving example.
- The existing cloudpickle warning is treated as an explicit artifact trust-boundary concern. This
  ticket does not blindly migrate the custom model bundle to `skops` without compatibility proof.

## Scope

- Refactor preprocessing so `FeaturePreprocessor.fit()` and `transform()` validate and select only
  `feature_columns`; target validation remains in training-specific functions.
- Keep strict missing-feature checks, stable feature ordering, unknown-category handling, and fitted
  transformer checks.
- Make Logistic Regression, Random Forest, and XGBoost wrappers accept feature-only frames in both
  `predict()` and `predict_proba()`.
- Preserve training failure when `is_fraud` is missing from a training frame.
- Build a representative, feature-only MLflow input example with deterministic dtypes matching the
  serving contract. Define nullable numeric behavior explicitly: encode potentially nullable
  numeric inputs as `float64`, or enforce/test non-nullability before retaining integer schema.
- Infer and require the MLflow signature before model logging/registration. Preserve the original
  exception cause and include an actionable error message.
- Log the model with the validated signature and feature-only input example.
- Load the newly logged model and run a prediction smoke check using the same feature-only example
  before moving the `candidate` alias.
- Ensure a signature, input-example, load, or prediction failure does not move the existing
  `candidate` alias. If registration has already created an immutable version before a later smoke
  failure, tag that version as rejected/validation-failed when practical and do not alias it.
- Record the feature contract version and signature/input-example validation outcome in model/run
  metadata.
- Document why cloudpickle artifacts must only be loaded from the trusted, access-controlled model
  registry.
- Add focused preprocessing, model-artifact, tracking-client, and SQLite-backed MLflow tests.

## Non-Scope

- Adding `is_fraud` to an inference request or MLflow input example.
- Lowering model quality gates or changing the selected Random Forest candidate.
- Automatic deletion of registered model version `2`; registry history should remain auditable.
- Automatic staging or production promotion.
- Switching to `skops` unless the complete custom preprocessing-plus-estimator bundle is proven
  compatible and round-trip tested. A serialization-format migration should otherwise be a separate
  security ticket.
- Building the FastAPI endpoint; that remains TRM-007 after this serving contract is valid.

## Acceptance Criteria

- A fitted model's `predict()` and `predict_proba()` succeed with a dataframe containing exactly the
  ordered pre-transaction feature columns and no target column.
- Training still fails clearly when its target column is absent, and inference still fails clearly
  when a required feature is absent.
- The generated MLflow signature contains model features only; `is_fraud`, identifiers, and
  forbidden post-event features are absent.
- The MLflow input example validates against the logged model and can be converted to serving input.
- Signature inference failure aborts before a model version or candidate alias is published.
- A post-registration load/prediction smoke failure does not move the existing candidate alias and
  remains traceable as a failed version/run.
- A successfully logged and reloaded model produces predictions of the expected shape from the
  feature-only input example.
- Integer/nullability behavior is either normalized or explicitly enforced and tested; the pipeline
  does not leave an unexplained schema warning for the supported serving contract.
- Tests cover feature-only transform/predict, missing training target, missing inference feature,
  signature contents, failure-before-publication, alias ordering, and MLflow round-trip prediction.
- Registered version `2` is documented as non-promotable; a later successful run may create a new
  version and move only the `candidate` alias to it.

## Suggested Validation

The repository owner runs Pytest. The implementing agent should add tests but limit its own
validation to static checks unless the owner changes that instruction.

```powershell
# Owner-executed tests
pytest tests/unit/test_preprocessing.py tests/unit/test_baseline.py
pytest tests/unit/test_model_artifact_contract.py tests/unit/test_tracking_client.py
pytest tests/integration/test_mlflow_training_run.py tests/integration/test_training_pipeline.py
make train

# Agent-executed static checks
ruff check .
ruff format --check .
python -m compileall -q app ml pipelines tests
git diff --check
```

## Prompt for an AI Agent

> Implement TRM-015 in the transaction-risk-ml repository. Read
> `docs/arsitektur_dan_tech_stack.md`, `docs/sprint_docs.md`, TRM-002, TRM-006, TRM-014, and this
> ticket before editing. Diagnose every training and inference caller of `FeaturePreprocessor` and
> separate feature-only selection from training-target extraction. All fitted model wrappers must
> predict from exactly the ordered pre-transaction feature contract without `is_fraud`, while
> training must still require the target. Build a deterministic feature-only MLflow input example,
> resolve or explicitly enforce integer/nullability semantics, and infer a required signature before
> publication. Do not catch and ignore signature failures. Log and reload the model, perform a
> feature-only prediction smoke check, and move the `candidate` alias only after that check succeeds.
> Preserve the existing alias on failure and keep any failed immutable version traceable. Do not add
> the target to serving input, do not change model quality thresholds, do not auto-promote, and do
> not replace cloudpickle with `skops` without a proven round-trip for the custom bundle. Add focused
> unit and SQLite-backed integration tests. The repository owner has reserved Pytest execution; do
> not run Pytest. Run Ruff, format check, compileall, and `git diff --check`, then report exact static
> evidence and remaining owner validation.
