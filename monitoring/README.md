# Runtime observability

`make up` starts the full local stack, including the API, Prometheus, Grafana, and containerized MLflow
tracking services. `make up-monitoring` remains available when only the API and dashboards are needed.
Both UIs bind to loopback only: Prometheus at `http://127.0.0.1:9090` and Grafana at
`http://127.0.0.1:3000`. Set `GRAFANA_ADMIN_PASSWORD` in `.env` before sharing a workstation session.
If an existing stack was created in Pod mode, follow the one-time migration instructions in
[`deployment/docker/README.md`](../deployment/docker/README.md) before starting this profile.

```powershell
make up-monitoring
podman compose ps
curl http://127.0.0.1:8000/metrics
```

The provisioned **Transaction Risk Runtime** Grafana dashboard displays prediction rate, p95 latency, 5xx
error percentage, readiness, and the active model name/version. The error panel shows 0% when the API is
scrapeable but there are no 5xx responses; it shows no data if the API cannot be scraped. The p95 panel
can have no data before the first prediction. Monitoring is
deliberately local and optional: an
unavailable Prometheus/Grafana service cannot block scoring, and the API's `/metrics` rendering/update
path isolates telemetry errors.

## Metric contract and privacy policy

| Metric | Unit | Labels | Purpose |
| --- | --- | --- | --- |
| `trm_http_requests_total` | requests | `method`, normalized `path`, `status` | Rate and error calculation |
| `trm_http_request_duration_seconds` | seconds | `method`, normalized `path` | p50/p95/p99 latency |
| `trm_http_in_flight_requests` | requests | `method`, normalized `path` | Concurrency/saturation signal |
| `trm_validation_rejections_total` | requests | normalized `path` | Public contract rejection rate |
| `trm_model_load_total` | attempts | `outcome` (`success`/`failure`) | Model startup failure detection |
| `trm_model_load_duration_seconds` | seconds | none | Startup/model-load time |
| `trm_model_ready` | boolean | none | Readiness state |
| `trm_model_info` | info | controlled model name/version/source/contract | Active model identification |
| `trm_prediction_decisions_total` | decisions | `decision` (`allow`/`review`) | Bounded business-flow visibility |
| `trm_prediction_events_total` | events | `outcome` (`queued`/`dropped`) | Event-queue pressure and loss visibility |
| `trm_prediction_event_store_failures_total` | attempts | none | Background event-store retries |
| `trm_delayed_label_ingestions_total` | labels | bounded `outcome` | Feedback endpoint visibility |

Paths are restricted to known routes plus `other`; status is an HTTP status; decision and model metadata
come from controlled application configuration/artifacts. Metrics must never include request IDs,
correlation IDs, account IDs, timestamps, amounts, raw payloads, user identifiers, exception text, or
query strings. Model versions are permitted only because one reviewed model is active per process; do not
add a user-supplied version label.

Set `LOG_JSON=true` to emit structured logs with `timestamp`, `level`, `logger`, `message`, and the
request correlation ID. The application never adds raw transaction payload fields to this event. Keep
correlation IDs in logs—not Prometheus labels—to prevent cardinality growth.

## Alert response runbooks

### Sustained 5xx errors or high p95 latency

1. Check `/health/ready`, `trm_model_ready`, in-flight requests, and CPU/memory.
2. Compare the traffic level with the committed load baseline before changing replica/resource settings.
3. If the regression followed a deployment, roll back the image/model revision; do not bypass readiness.

### No ready instance or model-load failure

1. Inspect API/container or pod events and the sanitized model-provider log.
2. Verify the model artifact and evaluation report are present, trusted, compatible, and match the
   configured paths.
3. Restore the last known-good image/model bundle. Readiness must remain false until loading validates.

Prometheus alerts are defined in `monitoring/prometheus/alerts.yml`. Grafana is visualization only;
alerting remains Prometheus-native for this local profile. For kind, scrape configuration/alert delivery
must be supplied by a production-like monitoring stack; a single-node kind cluster remains non-HA.

## Prediction events, delayed labels, and ML reports

Each successful prediction response includes a random `feedback_id`. It is unrelated to an account or
transaction identifier and is the only join key accepted by delayed-label ingestion. Event schema is
`prediction-event-v1`; label schema is `delayed-label-v1`.
The approved feature contract includes transaction amounts and balance values, so protect the local
SQLite volume and generated artifacts as sensitive financial data. Production storage must add encryption
at rest and access controls appropriate to the deployment.

```powershell
$feedbackId = 'copy-feedback-id-from-prediction-response'
$label = @{
  schema_version = 'delayed-label-v1'
  feedback_id = $feedbackId
  label = 1
  label_time = [DateTime]::UtcNow.ToString('o')
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/feedback/labels `
  -ContentType 'application/json' -Body $label
```

When API key authentication is enabled, add `-Headers @{ 'X-API-Key' = $env:API_KEY }`. A repeated
submission with the same label is idempotent; a different label for an already-labeled feedback ID
returns `409`. Labels can arrive before their queued event has reached SQLite. Labels with no matching
event and labels timestamped before the prediction are reported as unmatched/invalid; predictions older
than the configured seven-day label grace period without a label are reported as late-unlabeled.

SQLite is the local profile's durable store. The prediction handler only performs bounded in-memory
queue insertion; a background worker writes batches with retry. Queue-full events are dropped and exposed
by `trm_prediction_events_total{outcome="dropped"}`. A process crash can lose events still in memory;
the worker gets a bounded drain period on shutdown. Use an externally scheduled run to export reports:

```powershell
python -m pip install -e ".[monitoring]" -c requirements/constraints-py312.txt
make monitoring-report
```

When the API runs under Podman Compose, use its report-job image so it reads the same named SQLite
volume as the API:

```powershell
make monitoring-report-compose
```

The scheduled job reads the validation split as the reference, deterministically caps report samples,
and writes per-model/per-contract Evidently HTML/JSON plus `artifacts/monitoring/reports/latest-summary.json`.
The summary includes data quality, feature and score drift, calibration bins, classification metrics
when both label classes exist, unmatched labels, and late-unlabeled predictions. Old event and label
rows are deleted according to `ML_MONITORING_RETENTION_DAYS` (90 days by default); generated HTML/JSON
reports are not automatically deleted and should follow the workspace's artifact-retention policy.

To run it daily, configure Windows Task Scheduler or cron to execute `make monitoring-report-compose`
from the repository root after data/model artifacts are available. For a host-based API, schedule
`python -m pipelines.run_monitoring` from the repository root instead. The job is a separate process and
requires the API's event volume, processed validation split, model artifact, and evaluation report.
It cannot alter inference readiness or model aliases. `make down-clean` deletes the named `monitoring-data`
volume and therefore erases locally collected events and labels.
