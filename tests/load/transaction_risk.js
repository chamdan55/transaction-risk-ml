import http from "k6/http";
import { check, sleep } from "k6";
import { Counter, Rate } from "k6/metrics";

const baseUrl = (__ENV.BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const invalidPercent = Number(__ENV.INVALID_PERCENT || 5);
const apiKey = __ENV.K6_API_KEY;

const expectedResponses = new Counter("expected_responses");
const unexpectedResponses = new Counter("unexpected_responses");
const unexpectedResponseRate = new Rate("unexpected_response_rate");
const acceptedProtocolStatuses = http.expectedStatuses(200, 422);

export const options = {
  scenarios: {
    mixed_predictions: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: [
        { duration: __ENV.RAMP_UP || "30s", target: Number(__ENV.VUS || 8) },
        { duration: __ENV.HOLD || "60s", target: Number(__ENV.VUS || 8) },
        { duration: __ENV.RAMP_DOWN || "15s", target: 0 },
      ],
      gracefulRampDown: "10s",
    },
  },
  thresholds: {
    http_req_duration: ["p(95)<500", "p(99)<1000"],
    unexpected_response_rate: ["rate<0.01"],
  },
};

function requestId() {
  // The API requires UUID values and idempotency must not turn repeated load
  // requests into cache hits. VU + iteration give a deterministic unique value.
  const vu = (__VU >>> 0).toString(16).padStart(6, "0");
  const iteration = (__ITER >>> 0).toString(16).padStart(6, "0");
  return `00000000-0000-4000-8000-${vu}${iteration}`;
}

function payload(isInvalid) {
  const body = {
    transaction_type: "PAYMENT",
    amount: 100.0,
    origin_balance_before: 1000.0,
    destination_balance_before: 500.0,
    timestamp: "2026-01-01T10:00:00Z",
    request_id: requestId(),
  };
  if (isInvalid) {
    // A numeric string violates StrictFloat and must be rejected with 422.
    body.amount = "100.0";
  }
  return JSON.stringify(body);
}

export default function () {
  const isInvalid = ((__ITER + __VU) % 100) < invalidPercent;
  const headers = { "Content-Type": "application/json" };
  if (apiKey) headers["X-API-Key"] = apiKey;

  const response = http.post(`${baseUrl}/v1/predictions`, payload(isInvalid), {
    headers,
    responseCallback: acceptedProtocolStatuses,
  });
  const expectedStatus = isInvalid ? 422 : 200;
  const isExpected = response.status === expectedStatus;
  expectedResponses.add(isExpected ? 1 : 0);
  unexpectedResponses.add(isExpected ? 0 : 1);
  unexpectedResponseRate.add(!isExpected);
  check(response, { [`status is ${expectedStatus}`]: () => isExpected });
  sleep(Number(__ENV.THINK_TIME_SECONDS || 0));
}

export function handleSummary(data) {
  return {
    "artifacts/load/k6-summary.json": JSON.stringify(data, null, 2),
    stdout: "Load-test summary written to artifacts/load/k6-summary.json\n",
  };
}
