"""FastAPI application for synchronous transaction-risk scoring."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

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
from app.services.reliability import (
    IdempotencyConflictError,
    InMemoryIdempotencyStore,
    PredictionBusyError,
    PredictionExecutor,
    PredictionTimeoutError,
)

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
    prediction_executor = PredictionExecutor(
        max_concurrent_predictions=resolved_settings.max_concurrent_predictions,
        queue_timeout_seconds=resolved_settings.prediction_queue_timeout_seconds,
        prediction_timeout_seconds=resolved_settings.prediction_timeout_seconds,
    )
    idempotency_store = InMemoryIdempotencyStore(
        ttl_seconds=resolved_settings.idempotency_ttl_seconds,
        max_entries=resolved_settings.idempotency_max_entries,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved_provider.load()
        app.state.model_provider = resolved_provider
        app.state.prediction_executor = prediction_executor
        app.state.idempotency_store = idempotency_store
        try:
            yield
        finally:
            await prediction_executor.shutdown(resolved_settings.shutdown_grace_seconds)
            await idempotency_store.clear()

    app = FastAPI(
        title="Transaction Risk ML Platform",
        version="0.2.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def security_and_correlation_middleware(request: Request, call_next):
        correlation_id = _correlation_id(request.headers.get("x-request-id"))
        request.state.correlation_id = correlation_id
        if request.method == "POST":
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    is_too_large = int(content_length) > resolved_settings.max_request_bytes
                except ValueError:
                    return _error_response(400, "Invalid Content-Length", correlation_id)
                if is_too_large:
                    return _error_response(413, "Request body is too large", correlation_id)
            if len(await request.body()) > resolved_settings.max_request_bytes:
                return _error_response(413, "Request body is too large", correlation_id)
        if _requires_authentication(request.url.path) and not _is_authorized(
            request.headers.get("x-api-key"), resolved_settings
        ):
            return _error_response(401, "Unauthorized", correlation_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = str(correlation_id)
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.url.path == "/v1/predictions":
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        """Do not reflect a raw transaction payload in validation errors."""

        return _error_response(422, "Invalid request", _request_correlation_id(request))

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("Unhandled API error request_id=%s", _request_correlation_id(request))
        return _error_response(500, "Internal server error", _request_correlation_id(request))

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
    async def predict(
        payload: PredictionRequest, request: Request, response: Response
    ) -> PredictionResponse:
        provider = current_provider(request)
        if not provider.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not ready",
            )
        request_id = payload.request_id or _request_correlation_id(request)
        fingerprint = _request_fingerprint(payload)
        store = getattr(request.app.state, "idempotency_store", idempotency_store)
        try:
            owns_prediction, cached_response = await store.acquire(request_id, fingerprint)
        except IdempotencyConflictError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="request_id was already used with a different request",
            ) from None
        except PredictionBusyError:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Prediction capacity is temporarily unavailable",
            ) from None
        if not owns_prediction:
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(cached_response),
                    timeout=resolved_settings.prediction_timeout_seconds,
                )
            except TimeoutError:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Prediction timed out",
                ) from None
            except (Exception, asyncio.CancelledError):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Model is temporarily unavailable",
                ) from None
            response.headers["X-Idempotent-Replay"] = "true"
            LOGGER.info("prediction_replayed request_id=%s", request_id)
            return result

        try:
            feature_frame = prepare_feature_frame(payload)
            executor = getattr(request.app.state, "prediction_executor", prediction_executor)
            score = await executor.run(lambda: provider.predict_score(feature_frame))
        except PredictionBusyError as exc:
            await store.fail(request_id, exc)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Prediction capacity is temporarily unavailable",
            ) from None
        except PredictionTimeoutError as exc:
            await store.fail(request_id, exc)
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Prediction timed out",
            ) from None
        except ModelProviderError as exc:
            await store.fail(request_id, exc)
            LOGGER.exception("Prediction failed request_id=%s", request_id)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is temporarily unavailable",
            ) from None
        except Exception as exc:
            await store.fail(request_id, exc)
            LOGGER.exception("Unexpected prediction failure request_id=%s", request_id)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is temporarily unavailable",
            ) from None
        metadata = provider.metadata
        result = PredictionResponse(
            request_id=request_id,
            risk_score=score,
            decision="review" if score >= metadata.threshold else "allow",
            **metadata.__dict__,
        )
        await store.complete(request_id, result)
        LOGGER.info(
            "prediction_completed request_id=%s model_name=%s model_version=%s decision=%s",
            request_id,
            metadata.model_name,
            metadata.model_version,
            result.decision,
        )
        return result

    return app


app = create_app()


def _requires_authentication(path: str) -> bool:
    return path == "/model/info" or path.startswith("/v1/")


def _is_authorized(provided_key: str | None, settings: Settings) -> bool:
    if not settings.api_auth_enabled:
        return True
    if provided_key is None or settings.api_key is None:
        return False
    return hmac.compare_digest(provided_key, settings.api_key.get_secret_value())


def _correlation_id(value: str | None) -> UUID:
    if value is not None:
        try:
            return UUID(value)
        except ValueError:
            pass
    return uuid4()


def _request_correlation_id(request: Request) -> UUID:
    return getattr(request.state, "correlation_id", uuid4())


def _request_fingerprint(payload: PredictionRequest) -> str:
    body = payload.model_dump(mode="json", exclude={"request_id"})
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _error_response(status_code: int, detail: str, request_id: UUID) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail, "request_id": str(request_id)},
        headers={"X-Request-ID": str(request_id), "X-Content-Type-Options": "nosniff"},
    )
