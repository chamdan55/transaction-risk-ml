from __future__ import annotations

import json
import logging

from app.core.logging import JsonFormatter, reset_correlation_id, set_correlation_id


def test_json_logs_include_correlation_id_without_extra_request_fields() -> None:
    formatter = JsonFormatter()
    token = set_correlation_id("3b241101-e2bb-4255-8caf-4136c566a962")
    try:
        record = logging.LogRecord(
            name="app.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="prediction_completed model=random_forest",
            args=(),
            exc_info=None,
        )
        event = json.loads(formatter.format(record))
    finally:
        reset_correlation_id(token)

    assert event["correlation_id"] == "3b241101-e2bb-4255-8caf-4136c566a962"
    assert set(event) == {"timestamp", "level", "logger", "message", "correlation_id"}
