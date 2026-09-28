# Transaction Risk Serving Load Baseline

> Copy this template into a dated, reviewed report after running the commands. Do not commit API keys,
> raw transaction payloads, account identifiers, or other sensitive data.

## Environment

- Date/time and timezone:
- Commit/image digest:
- Model name/version and bundle digest:
- Runtime (Podman/OS/CPU/RAM):
- Deployment target (Compose or kind):
- API settings: max concurrent predictions, prediction timeout, authentication enabled/disabled:

## Workload and commands

- Base URL:
- k6 version:
- VUs, ramp-up, hold, ramp-down:
- Valid/invalid mix (invalid requests are expected `422`):
- Exact commands executed:

## Results

| Metric | Value |
| --- | --- |
| Requests / throughput | |
| Valid `200` | |
| Expected invalid `422` | |
| Unexpected responses / rate | |
| p50 latency | |
| p95 latency | |
| p99 latency | |
| CPU before/after/peak | |
| Memory before/after/peak | |
| Startup and model-load time | |

Attach or link the generated `k6-summary.json`, runtime statistics, and sanitized logs.

## Capacity decision

- Observed saturation point/bottleneck:
- Chosen replicas, CPU/memory requests/limits, and why:
- SLO gate retained or revised, and why:
- HPA decision (not added unless metric and saturation evidence exist):

## Resilience evidence

- Pod deletion result:
- Rolling update result and Ready endpoint observation:
- Bad-model readiness failure result:
- Rollback command/result:
- Known limitation: single-node kind does not provide host/node/disk/network HA.
