# TRM-007 — Build Shared Inference Contract and FastAPI Serving

**Type:** New capability
**Priority:** P0
**Sprint:** 4
**Dependencies:** TRM-002, TRM-003, TRM-006, TRM-014, TRM-015
**Status:** Implemented — awaiting owner validation

## Problem

The application currently exposes only basic health/config routes. There is no production-oriented
model loader, shared feature transformation, prediction endpoint, readiness state, or versioned
response contract.

## Scope

- Create strict request/response schemas for pre-transaction scoring.
- Reuse the versioned feature contract from training.
- Add a model provider that loads an explicit local artifact or approved registry alias at startup.
- Validate model signature, feature contract, threshold, and version before becoming ready.
- Implement `/v1/predictions`, `/health/live`, `/health/ready`, and `/model/info`.
- Keep the model in memory and avoid MLflow access per request.
- Return score, decision, model/version, threshold or policy version, contract version, and request ID.
- Add unit, contract, and end-to-end API tests including model-unavailable behavior.

## Non-Scope

- Authentication/rate limiting beyond extension hooks; completed in TRM-008.
- Online behavioral feature state.
- Automatic hot reload of a new model version.

## Acceptance Criteria

- API and training use exactly the same ordered feature contract.
- Valid requests produce deterministic responses for a fixed model.
- Unknown/missing/invalid fields are rejected.
- Liveness can be healthy while readiness is false if model loading fails.
- Startup fails safely or remains unready on schema/model mismatch.
- No per-request registry/model load occurs.

## Suggested Validation

```powershell
pytest tests/unit/test_health.py tests/contract tests/integration/test_api.py
uvicorn app.main:app
ruff check .
ruff format --check .
```

## Prompt for an AI Agent

> Implement TRM-007 only after its dependencies are complete. Read the architecture, sprint plan,
> feature contract, model manifest/signature, and current FastAPI skeleton. Build a lifespan-managed
> model provider and versioned prediction API that uses the exact training feature contract. Support
> a deterministic local-artifact mode for tests and an approved MLflow alias mode for integration.
> Keep loading out of the request path, distinguish liveness from readiness, and fail safely on
> contract mismatch. Add focused schema, loader, endpoint, and unavailable-model tests. Do not add
> Redis, Kubernetes, or monitoring infrastructure in this ticket. Preserve unrelated changes and
> report validation evidence.
