# TRM-010 — Establish Load Baseline and kind Deployment

**Type:** Performance/orchestration capability
**Priority:** P2
**Sprint:** 5
**Dependencies:** TRM-009
**Status:** Blocked

## Problem

Replica counts, resources, and autoscaling cannot be chosen responsibly without a measured service
capacity baseline. Local kind also must not be presented as host/node HA.

## Scope

- Add a reproducible load-test scenario with realistic valid/invalid request mix.
- Measure p50/p95/p99 latency, throughput, errors, CPU, memory, and startup/model-load time.
- Define initial service SLOs and capacity assumptions from evidence.
- Create kind manifests for namespace, Deployment, Service, ConfigMap, Secret references, probes,
  resources, rolling update, and PodDisruptionBudget.
- Add HPA only if a usable metric and saturation point are demonstrated.
- Test rollout, pod termination, model-load failure, and rollback scenarios.

## Non-Scope

- Multi-node or multi-region HA claim.
- Production cloud ingress.

## Acceptance Criteria

- Load report records environment, model, request mix, and reproducible commands.
- Resource settings trace back to measurements.
- At least one pod restart/rolling-update scenario preserves documented availability.
- Rollback to previous image/model is demonstrated.
- Documentation explicitly states kind single-node failure limitations.

## Suggested Validation

```powershell
pytest tests/load
kind create cluster
kubectl apply -f deployment/kubernetes
kubectl rollout status deployment/transaction-risk-api
kubectl rollout undo deployment/transaction-risk-api
```

## Prompt for an AI Agent

> Implement TRM-010 after the Compose/container runtime works. Add a reproducible load harness and
> first measure the API on fixed hardware; use those results to choose initial resources and any
> autoscaling policy. Then create minimal kind manifests with correct probe semantics, resources,
> rolling update, secret references, and rollback instructions. Exercise pod failure, bad model
> readiness, rollout, and rollback. Do not claim node or host HA from a single-node kind cluster.
> Store the load report and report all commands/results.
