# TRM-005 — Optimize Spark Split, Sampling, and Repeated Scans

**Type:** Performance and maintainability enhancement
**Priority:** P1
**Sprint:** 3.5 / Sprint 1 hardening
**Dependencies:** TRM-001
**Status:** Implemented in code — benchmark and full Spark validation pending runtime support

## Problem

The pipeline performs repeated actions and uses unpartitioned global `Window.orderBy` operations for
chronological split and deterministic sampling. Spark resources and output partition counts are
hard-coded.

## Scope

- Capture representative Spark physical plans and execution timings before changing behavior.
- Consolidate validation/count aggregations where possible.
- Cache/persist only reused DataFrames and unpersist deterministically.
- Replace global row-number sampling with deterministic hash-based logic where exact count is not a
  business requirement.
- Implement a scalable chronological split strategy while retaining deterministic boundaries and
  no-overlap guarantees.
- Move Spark master, memory, shuffle partitions, and output partitions into validated config.
- Clean up the feature-output directory layout.
- Add performance-regression or plan-shape assertions that are stable enough for CI.
- Add `pipelines.benchmark_spark` to capture elapsed time, split counts, and physical-plan evidence
  for exact versus time-boundary splitting.

## Non-Scope

- Provisioning a remote Spark cluster.
- Changing business feature semantics.

## Acceptance Criteria

- Split remains chronological, deterministic, exhaustive, and non-overlapping.
- Sampling remains deterministic and reports exact source/retained class counts.
- No unpartitioned global row-number window remains without an explicit documented justification.
- Reused pipeline stages are not recomputed unnecessarily.
- Resource/partition values are configurable with safe local defaults.
- Sample integration tests and full-data benchmark complete successfully.

## Suggested Validation

```powershell
pytest tests/unit/test_split.py tests/unit/test_sampling.py
pytest tests/integration/test_pipeline.py tests/integration/test_feature_split.py
python -m pipelines.run_pipeline
python -m pipelines.benchmark_spark
```

The benchmark and Spark integration commands require a Java/Spark runtime that can establish local
loopback connections. If that runtime is unavailable, report the benchmark as not executed.

## Prompt for an AI Agent

> Implement TRM-005 after TRM-001. Begin by inspecting Spark plans and identifying repeated actions,
> global windows, and hard-coded resource settings. Preserve chronological determinism and exact
> split isolation while replacing avoidable single-partition operations. Prefer measured changes
> over speculative rewrites, add config validation and focused tests, and provide before/after plan
> or timing evidence on the same dataset. Do not introduce Kafka, Airflow, or a remote cluster.
> Preserve unrelated work and report tests that could not run as not executed.
