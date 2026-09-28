# TRM-016 — Connect Training and Registry to Containerized MLflow

**Type:** MLOps infrastructure integration
**Priority:** P1
**Sprint:** 6.5 / prerequisite to Sprint 7
**Dependencies:** TRM-003, TRM-009, TRM-012
**Status:** Implemented — owner Podman/runtime validation pending

## Problem

The optional Podman Compose `tracking` profile already defines MLflow, PostgreSQL, and MinIO, but
the host-side `make train` and promotion commands still load `configs/tracking.yaml`, which points
to `sqlite:///mlflow.db` and the local `mlartifacts` directory. Starting `make up-tracking` alone
therefore does not move training runs or registry operations to the containerized server.

The current Compose server uses `--default-artifact-root s3://mlflow`, while MinIO has no host port.
Host-side clients must not be required to reach MinIO directly or possess its credentials. The
tracking client also always supplies an experiment `artifact_location`, so merely changing its URI
would retain the wrong local artifact destination for newly created remote experiments.

## Decision

- Keep Spark training on the host for this ticket; containerize the MLflow service, metadata store,
  and artifact store only. Do not add Spark to the serving image.
- Preserve the existing SQLite/local profile for lightweight tests and backward compatibility.
- Add an explicit container-tracking profile targeting the loopback MLflow endpoint. Use MLflow's
  proxied artifact mode (`--artifacts-destination` pointed at MinIO), so clients access artifacts
  through MLflow and MinIO stays private.
- Do not silently migrate or delete existing `mlflow.db`/`mlartifacts` data. Document the separate
  history and a deliberate migration decision if that history must be retained.

## Scope

1. Add PostgreSQL and S3-compatible storage drivers to a dedicated tracking-server image that
   excludes Spark and Java. Add actionable health checks for MLflow, PostgreSQL, and MinIO bucket
   initialization. Keep state in named volumes, expose only MLflow on host loopback, and do not log
   secrets. Validate `.env`/process settings before starting the profile. Keep the bucket initializer
   running after setup to avoid an exited one-shot container in the Podman project.
2. Change MLflow's server command to use `--artifacts-destination s3://mlflow` for proxied uploads
   and downloads, without an incompatible direct-client artifact root. Confirm the server can write
   and read from its private MinIO bucket under a read-only root filesystem.
3. Make `artifact_location` optional in the tracking config/client for server-managed remote
   experiments while preserving the local profile's explicit location. Introduce a separate,
   documented Compose tracking config with `http://127.0.0.1:${MLFLOW_PORT}`-equivalent host URI;
   account for a non-default host port without relying on `MLFLOW_TRACKING_URI` alone, because the
   client currently calls `mlflow.set_tracking_uri()` from YAML.
4. Provide unambiguous host commands for training and `pipelines.promote_model` against the same
   remote registry, plus a read-only connectivity check. Local commands should not accidentally
   publish to the remote registry; remote commands should fail clearly when the server is down.
5. Add focused tests for config selection, optional artifact location, URL/port handling,
   experiment creation, and failure behavior. Add a Podman integration smoke that creates a new
   remote experiment/run, uploads and downloads a small artifact, and confirms the registered model
   is visible in the remote registry. Keep this smoke separate from unit tests and avoid changing
   production aliases as part of the smoke.
6. Update `.env.example`, Makefile, deployment README, and architecture/sprint docs with startup,
   validation, persistence, backup/cleanup, and secret-handling instructions. The existing
   `make up-mlflow` local UI command must be clearly distinguished from `make up-tracking`.

## Non-Scope

- Containerizing Spark training, retraining, or the API's inference path.
- Migrating SQLite runs/models/artifacts automatically or deleting their history.
- Exposing PostgreSQL, MinIO, or an unauthenticated MLflow server beyond loopback.
- Automatic `staging`/`production` promotion, scheduled retraining, or rollout/rollback; those
  belong to TRM-013.
- Treating local Podman volumes as a production HA or backup solution.

## Acceptance Criteria

- `make up-tracking` yields a reachable MLflow UI/API and healthy stateful services; PostgreSQL
  and MinIO remain inaccessible from host ports.
- An explicit remote training invocation writes its run, metrics, artifacts, model version, and
  candidate alias to the containerized MLflow registry; the same run is visible in the UI.
- A host-side client can upload and download an artifact through MLflow without MinIO host access
  or MinIO credentials. The resulting experiment has a proxied artifact URI, not a host-local path.
- Remote promotion/registry commands use the same tracking configuration as remote training and
  retain existing human approval requirements. There is no implicit production promotion.
- The default SQLite configuration and its tests continue to work; old local history is preserved
  and is explicitly identified as separate from the new PostgreSQL history.
- Startup failure, missing dependencies, invalid config, and unavailable MLflow fail with useful
  diagnostics. No secrets are committed or printed in normal logs.
- Podman restart preserves metadata and artifacts in named volumes; destructive cleanup remains
  opt-in and documented. An end-to-end smoke validates artifact round-trip and model visibility.

## Suggested Validation

The repository owner runs Pytest. The implementing agent may add tests and run static checks, but
must leave Pytest execution to the owner unless that instruction changes.

```powershell
# Agent-executed static checks
ruff check .
ruff format --check .
git diff --check

# Owner-executed integration checks (use the exact commands added in this ticket)
make up-tracking
# Run the focused tracking-config/client tests and Podman artifact round-trip smoke.
# Run remote training, then inspect its experiment, artifacts, and model version in the MLflow UI.
# Restart the tracking profile and verify that metadata and artifacts are still present.
```

## Prompt for an AI Agent

> Implement TRM-016 before TRM-013. Read `docs/arsitektur_dan_tech_stack.md`,
> `docs/sprint_docs.md`, TRM-009, TRM-013, and this ticket; inspect the current Compose profile,
> MLflow client/config, training and promotion CLI, dependencies, and tests before editing. Keep
> host-side Spark training and the existing SQLite profile. Make the containerized MLflow +
> PostgreSQL + MinIO profile operational under Podman, with proxied artifact access and no MinIO
> host exposure. Add an explicit remote config and commands for both training and promotion;
> make experiment artifact location optional so the remote server chooses its proxied URI. Verify
> the remote run, artifact round-trip, and model registry visibility without automatically
> promoting to production. Preserve local history, named volumes, and unrelated user changes.
> Add focused tests and document the migration boundary, secrets, restart, and cleanup. The owner
> runs Pytest; do not run it. Report static-check results and exact owner-run Podman validation.
