# Podman and Compose Runtime

## Prerequisites

1. Podman with a Compose provider (`podman compose`). Podman Desktop is suitable on Windows.
2. A local serving artifact produced by `make train`:
   `artifacts/models/random_forest.joblib` and `artifacts/evaluation_report.json`.
3. Optionally copy `.env.example` to `.env`. Keep real API/database/object-store secrets only in
   `.env`, environment injection, or a secret manager; `.env` is ignored by Git.

## Full local stack

Copy `.env.example` to `.env` and set the required tracking secrets (`POSTGRES_PASSWORD`,
`MINIO_ROOT_USER`, and `MINIO_ROOT_PASSWORD`) before starting the full stack. The PostgreSQL password
must be 32–128 hexadecimal characters; see the example file for a generation command. Then run:

```powershell
make up
podman compose ps
curl http://127.0.0.1:8000/health/ready
make down
```

`make up` starts the API, Prometheus, Grafana, MLflow, PostgreSQL, MinIO, and the bucket initializer.
Prometheus/Grafana and MLflow are bound to loopback; PostgreSQL and MinIO remain private to the
Compose network. `monitoring-report` is a one-shot job and remains separate (`make monitoring-report-compose`).
For an API-only run, use `podman compose up --build -d api`; `make up-monitoring` starts the API and
dashboards without the tracking services, while `make up-tracking` starts only the tracking services.

The Compose API uses the `serving-with-model` image target, which packages the approved model and
evaluation report into the image at build time. The generic `serving` target remains independent of local
artifacts for CI, while the separate `model-bundle` target is used by kind. The API image contains the
installed wheel and has no source bind mount. The API runs as UID 10001,
with a read-only filesystem, dropped Linux capabilities, and only `/tmp` as tmpfs scratch space.

The Compose file explicitly disables pod mode for `podman-compose`. Services use the declared Compose
networks, and the API starts from an image that already contains its model bundle.

Older Compose revisions used a one-shot `model-seed` dependency. Podman could fail when starting
Prometheus/Grafana later because it tried to restart the already-completed seed. After upgrading from that
revision, stop the current Compose command, remove only this project's old containers, and recreate the
stack once:

```powershell
$env:PODMAN_COMPOSE_IN_POD = "false"
podman rm --force --ignore transaction-risk-ml_grafana_1 transaction-risk-ml_prometheus_1 transaction-risk-ml_api_1 transaction-risk-ml_model-seed_1
make up-monitoring
podman compose ps
```

This removes and recreates those four containers, so the API briefly stops. Named monitoring and tracking
volumes, images, and unrelated Podman containers are retained. The old `model-artifacts` volume is no
longer mounted and is not automatically deleted. Do not use `make down-clean` for this migration because
it deletes the active named volumes.

Set `API_AUTH_ENABLED=true` and a high-entropy `API_KEY` before starting if the API is accessible to
anything beyond a trusted local client. The Compose port is bound to `127.0.0.1`; a reverse proxy is
responsible for TLS and any external exposure.

## Tracking profile details

The full `make up` command enables this profile together with monitoring. To start only the tracking
services, set independent high-entropy values for `POSTGRES_PASSWORD`, `MINIO_ROOT_USER`, and
`MINIO_ROOT_PASSWORD` in `.env`, then run:

```powershell
make up-tracking
```

MLflow is available only on loopback at `http://127.0.0.1:5000` (or the configured `MLFLOW_PORT`).
PostgreSQL and MinIO are private Compose-network services with named volumes and no host ports. The
tracking image contains MLflow and its database/object-store drivers without the Spark training
runtime. Artifact uploads and downloads go through MLflow's artifact proxy to MinIO, so host-side
clients do not need MinIO credentials or network access. The server proxy setup follows the
[MLflow tracking-server guidance](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/).
The Compose tracking config forces host-side artifact uploads and downloads through MLflow. Recent
MLflow clients can otherwise follow presigned URLs directly to `minio:9000`, which is private to the
Compose network and unreachable from the host.
The MinIO server and client images come from `quay.io/minio`, since the Docker Hub image pull can
return an access-denied error. The bucket initializer uses a small BusyBox helper with the statically
linked `mc` binary because the upstream client image is a scratch image and does not include a shell.
It remains healthy after creating the bucket, so it does not leave an exited one-shot container in
the Podman project.

This is a persistent single-machine development profile, not an HA deployment or an automated
backup system. `make down` stops services and keeps their named volumes. `make down-clean` deletes
PostgreSQL, MinIO, Prometheus, and Grafana state; back up the MLflow database and MinIO objects
outside Podman before using it if the local history matters. Existing `mlflow.db` and `mlartifacts`
remain separate and are not imported into PostgreSQL/MinIO.

`make up-mlflow` still starts the host-local SQLite UI. It is separate from this containerized
tracking server. Likewise, plain `make train` continues using the local SQLite configuration.
Use the following commands for the containerized registry:

```powershell
make check-tracking-env
make check-tracking
make tracking-smoke
make train-tracking
make promote-tracking MODEL_VERSION=2 PROMOTION_STAGE=staging APPROVED_BY=reviewer PROMOTION_REASON="Passed reviewed quality gates"
```

`check-tracking` performs a read-only API query. `tracking-smoke` creates uniquely named experiment,
run, artifact, and registry records, then verifies an artifact upload/download round-trip; it does
not move a model alias. For a custom host port, set the same value in `.env` and pass it to each
Make target, for example `make check-tracking MLFLOW_PORT=5050`. Training and promotion must use
the same tracking URI and configuration. Do not point the API at the remote registry as part of
this ticket; serving model source remains independently configured.

Before relying on this profile, run a remote training job and verify the run, model version, and
candidate alias in the MLflow UI. Containerized tracking uses a separate PostgreSQL registry from
the existing local SQLite registry, so local model versions and aliases will not appear there.

## Cleanup

`make down` stops services and preserves named volumes. `make down-clean` additionally deletes the
PostgreSQL, MinIO, Prometheus, and Grafana volumes; this is destructive to their local state. The model
remains in the immutable API image and is rebuilt from the two local artifacts when needed.

## Image targets

```powershell
podman build --target serving -t transaction-risk-serving:local .
podman build --target serving-with-model -t transaction-risk-serving-bundled:local .
podman build --target training -t transaction-risk-training:local .
podman build --target tracking-server -t transaction-risk-tracking:local .
podman build --target minio-init -t transaction-risk-minio-init:local .
```

The serving target installs only the `serving` extra. The training target is intentionally larger
because it includes PySpark, a Java runtime, and MLflow. The dedicated tracking-server target
contains MLflow and its PostgreSQL/S3 drivers without PySpark or Java. The base image is a patch-level Python tag
supplied through `PYTHON_IMAGE`; CI/release automation can override it with a digest-pinned
reference.
