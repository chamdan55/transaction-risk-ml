# TRM-004 — Harden Packaging, Dependencies, and CI

**Type:** Engineering foundation enhancement
**Priority:** P1
**Sprint:** 3.5 / Sprint 0 and 3 hardening
**Dependencies:** None
**Status:** Implemented — local wheel verification passes; remote CI pending

## Problem

Setuptools package discovery includes only `app*`; editable installs can hide the absence of `ml`
and pipeline modules from a built wheel. Training and serving dependencies are also combined, and
there is no committed reproducibility strategy for resolved dependencies.

## Scope

- Ensure built distributions contain all intended Python packages.
- Add a wheel build and clean-install smoke test.
- Split dependency groups for shared/core, training, tracking, serving, monitoring, and development
  as far as current tooling reasonably supports.
- Establish and document a lock or constraints workflow for Python 3.12.
- Align MLflow version support with TRM-003.
- Make CI run lint, format, unit/integration tests, build, and clean-install smoke checks.
- Ensure CI failures are not hidden by broad skips.

## Non-Scope

- Docker image implementation.
- Cloud CI/CD deployment.

## Acceptance Criteria

- A built wheel contains `app`, `ml`, and required executable modules.
- Clean environment imports and a minimal app/model-contract smoke test pass without repository
  root on `PYTHONPATH`.
- Serving installation does not require PySpark unless explicitly selected.
- Dependency resolution is repeatable and documented.
- CI uses supported Python/dependency versions and exposes skipped tests.

## Suggested Validation

```powershell
python -m build
python -m pip install --force-reinstall dist/*.whl
pytest
ruff check .
ruff format --check .
```

## Prompt for an AI Agent

> Implement TRM-004. Inspect `pyproject.toml`, package layout, Makefile, and GitHub Actions before
> editing. Fix distribution discovery so clean wheel installs contain every required module. Design
> practical dependency extras separating serving from Spark/training/tracking, and add a documented
> Python 3.12 lock/constraints workflow. Extend CI with wheel build and clean-install smoke tests
> that do not rely on the repository root being importable. Coordinate the MLflow constraint with
> TRM-003 if it has landed; otherwise avoid conflicting edits and document the dependency. Preserve
> unrelated changes and report all validation evidence.
