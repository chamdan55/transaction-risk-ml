import logging
import sys
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from json import dumps

from app.core.config import get_settings

LOG_FORMAT = "| %(asctime)s | %(levelname)s | %(name)s | %(message)s"
CORRELATION_ID: ContextVar[str] = ContextVar("correlation_id", default="-")


class JsonFormatter(logging.Formatter):
    """Emit a small structured log event without request payload fields."""

    def format(self, record: logging.LogRecord) -> str:
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": CORRELATION_ID.get(),
        }
        if record.exc_info:
            event["exception"] = self.formatException(record.exc_info)
        return dumps(event, ensure_ascii=False)


def set_correlation_id(value: str) -> Token[str]:
    return CORRELATION_ID.set(value)


def reset_correlation_id(token: Token[str]) -> None:
    CORRELATION_ID.reset(token)


def setup_logging(log_level: str | None = None, *, json_logs: bool | None = None) -> None:
    """Configure the shared application and pipeline logging format."""
    settings = get_settings()
    use_json = settings.log_json if json_logs is None else json_logs

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter() if use_json else logging.Formatter(LOG_FORMAT))
    logging.basicConfig(
        level=(log_level or settings.log_level).upper(), handlers=[handler], force=True
    )
