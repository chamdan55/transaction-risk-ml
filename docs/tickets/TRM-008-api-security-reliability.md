# TRM-008 — Add API Security, Reliability, and Contract Tests

**Type:** Security/reliability enhancement
**Priority:** P1
**Sprint:** 4
**Dependencies:** TRM-007
**Status:** Implemented — awaiting owner validation

## Problem

A prediction endpoint handling transaction data needs explicit authentication boundaries, input
limits, safe logging, timeouts, idempotency behavior, and concurrency verification.

## Scope

- Add a configurable authentication mechanism suitable for internal/local deployment.
- Add request-size and concurrency limits plus bounded prediction timeout.
- Define idempotency behavior for repeated transaction/request IDs.
- Produce structured logs with correlation ID while redacting account identifiers and payload data.
- Remove or restrict the current `/config` endpoint.
- Add graceful startup/shutdown and exception mapping without leaking internals.
- Add negative security tests, concurrency tests, and OpenAPI contract assertions.
- Document TLS termination and secret-injection responsibilities at the reverse-proxy/deployment
  boundary.

## Non-Scope

- Building an enterprise identity provider.
- Persisting full prediction events; handled by TRM-012.

## Acceptance Criteria

- Unauthorized requests are rejected in secured mode.
- Secrets and raw account identifiers never appear in responses or logs.
- Oversized, malformed, duplicate, and timed-out requests have documented behavior.
- Concurrent predictions do not mutate shared model state.
- Health endpoints remain usable by orchestrator probes under documented rules.
- Security and API contract tests pass.

## Suggested Validation

```powershell
pytest tests/contract tests/integration/test_api.py tests/integration/test_api_security.py
ruff check .
ruff format --check .
```

## Prompt for an AI Agent

> Implement TRM-008 on top of the completed prediction API. Add a minimal configurable internal
> authentication layer, request/concurrency limits, timeout behavior, correlation IDs, idempotency
> semantics, safe error handling, and privacy-aware structured logging. Restrict or remove `/config`.
> Keep TLS and secret injection at a documented deployment boundary instead of inventing a full
> identity platform. Add negative, concurrency, and OpenAPI contract tests, and verify shared model
> state is read-only during prediction. Preserve unrelated changes and report exact tests/results.
