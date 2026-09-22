# TRM-009 — Build Minimal Images and Docker Compose Runtime

**Type:** Deployment capability
**Priority:** P1
**Sprint:** 5
**Dependencies:** TRM-004, TRM-008
**Status:** Blocked

## Problem

The project has no usable Docker image or reproducible local service topology. A single image with
Spark, training, tracking, and serving dependencies would be unnecessarily large and risky.

## Scope

- Create separate minimal serving and training image targets.
- Use pinned base image/digest strategy, multi-stage builds, non-root user, and healthcheck.
- Ensure serving image installs built artifacts rather than bind-mounting source.
- Add Docker Compose profiles for the minimum API demo and optional MLflow/PostgreSQL/MinIO stack.
- Configure volumes, networks, secrets/env files, and startup dependencies safely.
- Add image smoke tests, vulnerability scanning, and SBOM generation in CI.
- Add `make up`, `make down`, and documented clean-reset behavior.

## Non-Scope

- Kubernetes resources.
- Cloud registry publication unless separately authorized.

## Acceptance Criteria

- Serving image excludes PySpark and training-only packages.
- Container runs as non-root and becomes ready using an immutable model artifact/reference.
- Compose starts a working prediction demo with one command.
- Stateful services use named volumes and are not exposed publicly by default.
- Image smoke test, scan policy, and SBOM complete in CI.

## Suggested Validation

```powershell
docker build --target serving .
docker compose up --build
docker compose ps
pytest tests/integration/test_container_smoke.py
```

## Prompt for an AI Agent

> Implement TRM-009 after packaging and API hardening are complete. Build distinct training and
> minimal serving image targets; the serving image must install the built package and exclude Spark
> and development tooling. Run as non-root, add a useful healthcheck, and prefer a read-only root
> filesystem. Create Compose profiles for a minimal API and optional MLflow/PostgreSQL/MinIO stack
> with private defaults and persistent volumes. Add CI smoke, scan, and SBOM steps without pushing
> images externally. Update Makefile/docs and report build size, startup behavior, and validation.
