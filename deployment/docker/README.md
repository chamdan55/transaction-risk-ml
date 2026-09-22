# Podman and Compose Runtime

## Prerequisites

1. Podman with a Compose provider (`podman compose`). Podman Desktop is suitable on Windows.
2. A local serving artifact produced by `make train`:
   `artifacts/models/random_forest.joblib` and `artifacts/evaluation_report.json`.
3. Optionally copy `.env.example` to `.env`. Keep real API/database/object-store secrets only in
   `.env`, environment injection, or a secret manager; `.env` is ignored by Git.

## Minimum local API

```powershell
make up
podman compose ps
curl http://127.0.0.1:8000/health/ready
make down
```

`model-seed` copies the explicitly built model artifact into the `model-artifacts` named volume.
The API image contains the installed wheel and has no source bind mount. The API runs as UID 10001,
with a read-only filesystem, dropped Linux capabilities, and only `/tmp` as tmpfs scratch space.

Set `API_AUTH_ENABLED=true` and a high-entropy `API_KEY` before starting if the API is accessible to
anything beyond a trusted local client. The Compose port is bound to `127.0.0.1`; a reverse proxy is
responsible for TLS and any external exposure.

## Optional tracking profile

Set `POSTGRES_PASSWORD`, `MINIO_ROOT_USER`, and `MINIO_ROOT_PASSWORD` in `.env`, then run:

```powershell
make up-tracking
```

MLflow is available only on loopback at `http://127.0.0.1:5000`. PostgreSQL and MinIO are private
Compose-network services with named volumes; they have no host ports.

## Cleanup

`make down` stops services and preserves named volumes. `make down-clean` additionally deletes
`model-artifacts`, `postgres-data`, and `minio-data`; this is destructive and requires retraining or
re-seeding the model before the next API start.

## Image targets

```powershell
podman build --target serving -t transaction-risk-serving:local .
podman build --target training -t transaction-risk-training:local .
```

The serving target installs only the `serving` extra. The training target is intentionally larger
because it includes PySpark, a Java runtime, and MLflow. The base image is a patch-level Python tag
supplied through `PYTHON_IMAGE`; CI/release automation can override it with a digest-pinned
reference.
