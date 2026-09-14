import logging
import sys

from app.core.config import get_settings

LOG_FORMAT = "| %(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(log_level: str | None = None) -> None:
    """Configure the shared application and pipeline logging format."""
    settings = get_settings()

    logging.basicConfig(
        level=(log_level or settings.log_level).upper(),
        format=LOG_FORMAT,
        stream=sys.stdout,
        force=True,
    )
