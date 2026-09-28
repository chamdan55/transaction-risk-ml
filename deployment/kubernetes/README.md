# kind deployment and load baseline

This directory is a local, production-like deployment demonstration. It is **not** a high-availability
environment: a single-node kind cluster cannot survive a host, node, disk, Podman VM, or local-network
failure. Two replicas, probes, a rolling update, and the PodDisruptionBudget only demonstrate resilience
to an individual process/pod failure while that one node remains healthy.

## Prerequisites

- Podman running and the API image buildable with `podman build`.
- `kind`, `kubectl`, and [k6](https://grafana.com/docs/k6/latest/set-up/install-k6/).
- On Podman, select its experimental kind provider for the current PowerShell session:

```powershell
$env:KIND_EXPERIMENTAL_PROVIDER = "podman"
```

The Podman provider depends on the local versions of Podman and kind. Treat a provider failure as an
environment issue, not as a Kubernetes availability result.

## Create the reproducible local deployment

Build separate serving and immutable model-bundle images, then load both into kind. The application image
does not carry the model artifact; the init container copies the reviewed bundle to a pod-local `emptyDir`
before the non-root API starts.

```powershell
podman build --target serving --tag transaction-risk-serving:local .
podman build --target model-bundle --tag transaction-risk-model-bundle:local .
kind create cluster --name transaction-risk --config deployment/kubernetes/kind-config.yaml

$servingTar = Join-Path $env:TEMP "trm-serving-kind.tar"
podman tag localhost/transaction-risk-serving:local docker.io/library/transaction-risk-serving:local
podman save --format docker-archive -o $servingTar docker.io/library/transaction-risk-serving:local
kind load image-archive $servingTar --name transaction-risk

$modelBundle = Join-Path $env:TEMP "trm-modelBundle-kind.tmp"
podman tag localhost/transaction-risk-model-bundle:local docker.io/library/transaction-risk-model-bundle:local
podman save --format docker-archive -o $modelBundle docker.io/library/transaction-risk-model-bundle:local
kind load image-archive $modelBundle --name transaction-risk

kubectl apply -k deployment/kubernetes
kubectl -n transaction-risk rollout status deployment/transaction-risk-api --timeout=3m
kubectl -n transaction-risk get pods,svc,pdb
```

For an authenticated deployment, create the real secret outside source control and set
`API_AUTH_ENABLED: "true"` in a reviewed environment-specific ConfigMap overlay. The example Secret is
intentionally not included in `kustomization.yaml`.

```powershell
kubectl -n transaction-risk create secret generic transaction-risk-api-secrets --from-literal=API_KEY='replace-with-a-secret'
```

Use a second terminal to make the ClusterIP service reachable locally:

```powershell
kubectl -n transaction-risk port-forward service/transaction-risk-api 8080:80
Invoke-WebRequest http://127.0.0.1:8080/health/ready
```

## Baseline protocol

Do not present the provisional resource values in `deployment.yaml` as measured capacity. Before changing
replicas, requests/limits, or adding an HPA, collect a baseline on fixed hardware and commit a completed
copy of `tests/load/baseline-report-template.md` (with no secrets) alongside the change that adjusts those
values.

Start the Compose API or port-forward the kind Service, wait until `/health/ready` is 200, then run the
same workload and metadata collection each time:

```powershell
New-Item -ItemType Directory -Force artifacts/load | Out-Null
podman stats --no-stream --format json transaction-risk-api-1 | Out-File -Encoding utf8 artifacts/load/podman-stats-before.json
$env:BASE_URL = "http://127.0.0.1:8000"
$env:VUS = "8"
$env:RAMP_UP = "30s"
$env:HOLD = "60s"
$env:RAMP_DOWN = "15s"
$env:INVALID_PERCENT = "5"
k6 run tests/load/transaction_risk.js
podman stats --no-stream --format json transaction-risk-api-1 | Out-File -Encoding utf8 artifacts/load/podman-stats-after.json
```

For kind, use `kubectl top pod -n transaction-risk` immediately before and after the k6 run (Metrics
Server is required). Record startup/model-load time as the duration from `kubectl apply -k ...` to the
first successful `/health/ready`. The load scenario treats valid requests (`200`) and intentional schema
rejections (`422`) as expected; any other status is an unexpected response. Give `K6_API_KEY` only through
the process environment when API authentication is enabled.

The initial SLO proposal is p95 below 500 ms, p99 below 1 s, and under 1% unexpected responses at the
recorded workload. These are test gates, not a production capacity claim; revise them only with the
attached evidence. No HPA is deployed because neither a usable metric pipeline nor a measured saturation
point exists yet.

## Failure, rollout, and rollback evidence

Run and record each command/result in the completed baseline report:

```powershell
# Pod/process failure: the replacement pod must become Ready.
kubectl -n transaction-risk delete pod -l app.kubernetes.io/name=transaction-risk-api
kubectl -n transaction-risk rollout status deployment/transaction-risk-api --timeout=3m

# A regular rollout must retain one Ready endpoint (maxUnavailable: 0).
kubectl -n transaction-risk rollout restart deployment/transaction-risk-api
kubectl -n transaction-risk rollout status deployment/transaction-risk-api --timeout=3m

# Demonstrate model-load/readiness failure, then restore the previous revision.
kubectl -n transaction-risk set env deployment/transaction-risk-api MODEL_ARTIFACT_PATH=/models/missing.joblib
kubectl -n transaction-risk rollout status deployment/transaction-risk-api --timeout=90s
kubectl -n transaction-risk rollout undo deployment/transaction-risk-api
kubectl -n transaction-risk rollout status deployment/transaction-risk-api --timeout=3m
```

The intentionally bad-model rollout is expected to time out because new pods remain NotReady; it must be
followed by `rollout undo`. `maxUnavailable: 0` preserves the prior Ready replica set while the bad revision
is unavailable. This is pod-level behavior only, not node/host HA.

## Controlled model release

TRM-013 exports an explicitly promoted `production` alias into an immutable versioned bundle. After
`make export-production-model`, pass the exact output directory to `make kind-rollout`; the command
builds an image tag containing both the registry version and bundle digest, saves it as a temporary
Docker-format archive with Podman, loads that archive into kind without requiring the Docker CLI,
updates the model-bundle init image and model path in one Deployment revision, and waits for readiness.
If the new pods do not become Ready, it rolls the Deployment back and restores the previous MLflow
production alias. The first production version has no rollback target, so establish a known-good
production version before using this automated release path.

## Scheduled retraining

`retraining-cronjob.yaml` is part of the Kustomize resources but has `suspend: true`. It cannot run
until operators provision the training-data PVC, the approved-manifest ConfigMap, and the Secret
`transaction-risk-retraining-secrets` with key `tracking-uri` and optional `tracking-username`,
`tracking-password`, or `tracking-token` keys; the training image must also be loaded as
`transaction-risk-training:local`. Point the tracking URI at an endpoint reachable from kind, using
HTTPS and authentication when outside a local demo. Review the gates in `configs/retraining.yaml`
before enabling the schedule; their committed values are provisional PaySim demonstration limits.

Update the ConfigMap only with a newly reviewed, content-addressed manifest, then unsuspend the CronJob.
Each job has concurrency `Forbid`, no automatic retry, a six-hour deadline, and a seven-day TTL for
completed Job objects. It consumes the same approved manifest until an operator changes it; drift
alerts do not create manifests, launch retraining, or promote a candidate.

Clean up the local cluster when finished:

```powershell
kind delete cluster --name transaction-risk
```
