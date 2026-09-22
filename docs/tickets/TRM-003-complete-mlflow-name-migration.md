# TRM-003 — Complete MLflow 3.x `name` Migration

**Type:** Defect fix
**Priority:** P0
**Sprint:** 3.5 / Sprint 3 reopening
**Dependencies:** None
**Status:** Implemented — targeted unit and SQLite integration tests pass

## Problem

The MLflow client intentionally calls `mlflow.sklearn.log_model(..., name=...)`, but the migration
is inconsistent. `LoggedModel` still defines `artifact_path`, its constructor is called with
`name`, and the unit-test fake still accepts `artifact_path`. Registry unit and integration tests
fail.

## Scope

- Keep the intentional MLflow 3.x external API use of `name`.
- Decide whether the internal domain field remains `artifact_path` or is renamed; apply the decision
  consistently across implementation, serialization, CLI, and tests.
- Update test doubles to match the supported external API.
- Set a dependency floor compatible with the chosen MLflow API and test that range in CI.
- Restore SQLite-backed model registration, load, alias, and promotion tests.
- Add model signature and a safe representative input example where practical.

## Non-Scope

- Changing model features or retraining the full dataset.
- Deploying a remote MLflow server.

## Acceptance Criteria

- Registry unit test passes.
- Both existing MLflow integration tests pass.
- Registered model can be loaded and promoted via explicit alias workflow.
- No ambiguous mix of `name` and internal field names remains.
- Declared MLflow dependency range matches the tested API.
- Sprint 3 documentation reflects evidence rather than assumed status.

## Suggested Validation

```powershell
pytest tests/unit/test_tracking_client.py tests/unit/test_tracking_registry.py
pytest tests/integration/test_mlflow_training_run.py
ruff check .
ruff format --check .
```

## Prompt for an AI Agent

> Fix TRM-003 without reverting the intentional use of MLflow 3.x `name`. Read the architecture,
> sprint plan, this ticket, `ml/tracking/client.py`, `ml/tracking/registry.py`, and all tracking
> tests. Make the external MLflow API, internal `LoggedModel` representation, fake clients, and
> dependency constraints consistent. Preserve explicit human-gated promotion. Add or improve model
> signature/input-example coverage if supported by the existing bundle. Run registry unit tests and
> real SQLite MLflow integration tests. Do not mark full Sprint 3 complete unless the relevant tests
> actually run and pass; report any environment blocker precisely.
