# TRM-001 — Define Pre-Transaction Feature Contract

**Type:** Correctness enhancement
**Priority:** P0
**Sprint:** 3.5 / Sprint 1 hardening
**Dependencies:** None
**Status:** Implemented — Spark-backed validation remains environment-dependent

## Problem

The current model feature set includes balances and derivatives that are only known after a
transaction is processed. The primary system contract is now pre-transaction risk scoring.

## Scope

- Introduce one versioned feature contract shared by data preparation, training, and future serving.
- Separate canonical/audit columns from online model features.
- Exclude direct and derived post-event balance features.
- Define feature names, types, order, nullability, availability phase, and semantic description.
- Update dataset loader and config validation to consume this contract instead of deriving model
  features indirectly from the complete feature schema.
- Add tests that fail if target, proxy label, IDs, future data, or post-event derivatives enter the
  online feature set.
- Document how historical features are treated before an online state source exists.

## Non-Scope

- Retraining models.
- Building the FastAPI endpoint.
- Adding Redis or a feature-store product.

## Acceptance Criteria

- A single importable contract defines the ordered pre-transaction model features.
- `origin_balance_after`, `destination_balance_after`, and all derivatives requiring them are not
  model inputs.
- Canonical data may retain post-event fields for audit/profiling.
- Training dataset validation uses the contract.
- Direct and derived leakage tests pass.
- Architecture, feature documentation, and config examples are consistent.

## Suggested Validation

```powershell
pytest tests/unit/test_feature_schema.py tests/unit/test_feature_leakage.py
pytest tests/unit/test_training_dataset.py
ruff check .
ruff format --check .
```

## Prompt for an AI Agent

> Implement ticket TRM-001 in the transaction-risk-ml repository. First read
> `docs/arsitektur_dan_tech_stack.md`, `docs/sprint_docs.md`, and this ticket, then inspect the
> existing feature schema, feature builders, split constants, model config, dataset loader, and
> leakage tests. Create a versioned pre-transaction feature contract that is the sole source of
> truth for training and future serving. Keep post-event balances in canonical/audit data but
> exclude them and every derivative requiring them from model inputs. Add focused tests for direct
> and derived leakage, preserve unrelated user changes, and update only documentation/configuration
> affected by the implementation. Do not retrain models in this ticket. Run the listed tests plus
> relevant existing tests; report exactly what ran, what passed, and any environment blocker.
