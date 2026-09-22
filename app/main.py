"""FastAPI application for synchronous transaction-risk scoring."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status

from app.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.core.config import Settings, get_settings
from app.core.logging import setup_logging
from app.services.feature_preparation import prepare_feature_frame
from app.services.model_provider import ModelProvider, ModelProviderError

setup_logging()
LOGGER = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    provider: ModelProvider | None = None,
) -> FastAPI:
    """Create an app whose model is loaded once during its lifespan."""

    resolved_settings = settings or get_settings()
    resolved_provider = provider or ModelProvider(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved_provider.load()
        app.state.model_provider = resolved_provider
        yield

    app = FastAPI(
        title="Transaction Risk ML Platform",
        version="0.2.0",
        lifespan=lifespan,
    )

    def current_provider(request: Request) -> ModelProvider:
        return getattr(request.app.state, "model_provider", resolved_provider)

    @app.get("/health/live", response_model=HealthResponse, response_model_exclude_none=True)
    def health_live() -> HealthResponse:
        return HealthResponse(status="live")

    @app.get("/health", response_model=HealthResponse, response_model_exclude_none=True)
    def health_legacy() -> HealthResponse:
        """Compatibility endpoint retained for the original application skeleton."""

        return HealthResponse(status="ok")

    @app.get("/health/ready", response_model=HealthResponse, response_model_exclude_none=True)
    def health_ready(request: Request) -> HealthResponse:
        provider = current_provider(request)
        if provider.is_ready:
            return HealthResponse(status="ready")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=HealthResponse(status="not_ready").model_dump(exclude_none=True),
        )

    @app.get("/model/info", response_model=ModelInfoResponse)
    def model_info(request: Request) -> ModelInfoResponse:
        provider = current_provider(request)
        if not provider.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not ready",
            )
        metadata = provider.metadata
        return ModelInfoResponse(status="ready", **metadata.__dict__)

    @app.post("/v1/predictions", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        provider = current_provider(request)
        if not provider.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not ready",
            )
        try:
            score = provider.predict_score(prepare_feature_frame(payload))
        except ModelProviderError:
            LOGGER.exception("Prediction failed for a ready model")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is temporarily unavailable",
            ) from None
        metadata = provider.metadata
        return PredictionResponse(
            request_id=payload.request_id or uuid4(),
            risk_score=score,
            decision="review" if score >= metadata.threshold else "allow",
            **metadata.__dict__,
        )

    @app.get("/config")
    def config() -> dict[str, str]:
        """Non-sensitive runtime selection metadata for local diagnostics."""

        return {
            "environment": resolved_settings.environment,
            "model_name": resolved_settings.model_name,
            "model_version": resolved_settings.model_version,
            "model_source": resolved_settings.model_source,
        }

    return app


app = create_app()
