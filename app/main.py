import logging

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title="Transaction Risk ML Platform",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    logger.info("Health check requested")

    return {"status": "ok"}


@app.get("/config")
def config() -> dict[str, str]:
    logger.info("Configuration requested")

    return {
        "environment": settings.environment,
        "model_name": settings.model_name,
        "model_version": settings.model_version,
    }
