"""FastAPI application for synchronous transaction-risk scoring."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from time import perf_counter
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.schemas import (
    DelayedLabelRequest,
    DelayedLabelResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.core.config import Settings, get_settings
from app.core.logging import reset_correlation_id, set_correlation_id, setup_logging
from app.services.feature_preparation import prepare_feature_frame
from app.services.metrics import METRICS_CONTENT_TYPE, RuntimeMetrics
from app.services.model_provider import ModelProvider, ModelProviderError
from app.services.reliability import (
    IdempotencyConflictError,
    InMemoryIdempotencyStore,
    PredictionBusyError,
    PredictionExecutor,
    PredictionTimeoutError,
)
from ml.monitoring.contracts import DelayedLabel, PredictionEvent
from ml.monitoring.store import LabelConflictError, PredictionEventStore

setup_logging()
LOGGER = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    provider: ModelProvider | None = None,
    event_store: PredictionEventStore | None = None,
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
    metrics = RuntimeMetrics()
    resolved_event_store = event_store or PredictionEventStore(
        resolved_settings.ml_monitoring_database_path,
        queue_maxsize=resolved_settings.ml_monitoring_queue_maxsize,
        retention_days=resolved_settings.ml_monitoring_retention_days,
        on_write_failure=metrics.record_prediction_event_store_failure,
        on_shutdown_drop=lambda count: metrics.record_prediction_event("dropped", count=count),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved_event_store.start()
        load_started_at = perf_counter()
        resolved_provider.load()
        app.state.model_provider = resolved_provider
        app.state.prediction_executor = prediction_executor
        app.state.idempotency_store = idempotency_store
        app.state.prediction_event_store = resolved_event_store
        metrics.record_model_load(
            ready=resolved_provider.is_ready,
            duration_seconds=perf_counter() - load_started_at,
        )
        metrics.set_model_metadata(
            resolved_provider.metadata if resolved_provider.is_ready else None
        )
        try:
            yield
        finally:
            await resolved_event_store.close(resolved_settings.shutdown_grace_seconds)
            await prediction_executor.shutdown(resolved_settings.shutdown_grace_seconds)
            await idempotency_store.clear()

    app = FastAPI(
        title="Transaction Risk ML Platform",
        version="0.2.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def security_and_correlation_middleware(request: Request, call_next):
        metric_path = _metric_path(request)
        if metric_path is not None:
            metrics.begin_request(request.method, metric_path)
        started_at = perf_counter()
        correlation_id = _correlation_id(request.headers.get("x-request-id"))
        request.state.correlation_id = correlation_id
        correlation_token = set_correlation_id(str(correlation_id))
        response: Response | None = None
        try:
            if request.method == "POST":
                content_length = request.headers.get("content-length")
                if content_length is not None:
                    try:
                        is_too_large = int(content_length) > resolved_settings.max_request_bytes
                    except ValueError:
                        response = _error_response(400, "Invalid Content-Length", correlation_id)
                        return response
                    if is_too_large:
                        response = _error_response(413, "Request body is too large", correlation_id)
                        return response
                if len(await request.body()) > resolved_settings.max_request_bytes:
                    response = _error_response(413, "Request body is too large", correlation_id)
                    return response
            if _requires_authentication(request.url.path) and not _is_authorized(
                request.headers.get("x-api-key"), resolved_settings
            ):
                response = _error_response(401, "Unauthorized", correlation_id)
                return response
            response = await call_next(request)
            response.headers["X-Request-ID"] = str(correlation_id)
            response.headers["X-Content-Type-Options"] = "nosniff"
            if request.url.path == "/v1/predictions":
                response.headers["Cache-Control"] = "no-store"
            return response
        finally:
            reset_correlation_id(correlation_token)
            if metric_path is not None:
                metrics.end_request(request.method, metric_path)
                metrics.observe_request(
                    request.method,
                    metric_path,
                    response.status_code if response is not None else 500,
                    started_at,
                )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        """Do not reflect a raw transaction payload in validation errors."""

        metrics.record_validation_rejection(_metric_path(request) or "other")
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
        metrics.set_model_metadata(provider.metadata if provider.is_ready else None)
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

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(content=metrics.render(), media_type=METRICS_CONTENT_TYPE)

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

        inference_started_at = perf_counter()
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
        feedback_id = uuid4()
        result = PredictionResponse(
            request_id=request_id,
            feedback_id=feedback_id,
            risk_score=score,
            decision="review" if score >= metadata.threshold else "allow",
            **metadata.__dict__,
        )
        try:
            feature_values = {
                name: _feature_scalar(value)
                for name, value in feature_frame.iloc[0].to_dict().items()
            }
            event = PredictionEvent(
                feedback_id=feedback_id,
                event_time=datetime.now(UTC),
                model_name=metadata.model_name,
                model_version=metadata.model_version,
                feature_contract_version=metadata.feature_contract_version,
                features=feature_values,
                risk_score=score,
                decision_threshold=metadata.threshold,
                decision=result.decision,
                latency_seconds=perf_counter() - inference_started_at,
            )
            queued = resolved_event_store.enqueue(event)
            metrics.record_prediction_event("queued" if queued else "dropped")
            if not queued:
                LOGGER.warning("Prediction event queue is full; event dropped")
        except Exception as exc:
            # Event telemetry stays outside the synchronous scoring availability boundary.
            metrics.record_prediction_event("dropped")
            LOGGER.warning("Unable to enqueue prediction event; error_type=%s", type(exc).__name__)
        await store.complete(request_id, result)
        metrics.record_decision(result.decision)
        LOGGER.info(
            "prediction_completed request_id=%s model_name=%s model_version=%s decision=%s",
            request_id,
            metadata.model_name,
            metadata.model_version,
            result.decision,
        )
        return result

    @app.post(
        "/v1/feedback/labels",
        response_model=DelayedLabelResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def ingest_delayed_label(payload: DelayedLabelRequest) -> DelayedLabelResponse:
        label = DelayedLabel(**payload.model_dump())
        try:
            duplicate = await resolved_event_store.record_label(label)
        except LabelConflictError:
            metrics.record_delayed_label("conflict")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="feedback_id already has a different label",
            ) from None
        except Exception:
            metrics.record_delayed_label("unavailable")
            LOGGER.exception("Unable to persist delayed label")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Feedback storage is temporarily unavailable",
            ) from None
        metrics.record_delayed_label("duplicate" if duplicate else "accepted")
        return DelayedLabelResponse(
            feedback_id=payload.feedback_id,
            status="duplicate" if duplicate else "accepted",
        )

    return app


app = create_app()


def _requires_authentication(path: str) -> bool:
    return path == "/model/info" or path.startswith("/v1/")


def _metric_path(request: Request) -> str | None:
    """Map requests to a deliberately small set of route labels."""

    path = request.url.path
    if path == "/metrics":
        return None
    if path in {
        "/health",
        "/health/live",
        "/health/ready",
        "/model/info",
        "/v1/predictions",
        "/v1/feedback/labels",
    }:
        return path
    return "other"


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


def _feature_scalar(value: object) -> str | int | float:
    """Convert pandas/numpy scalars to JSON-safe contract values."""

    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return value
    raise ValueError("Feature contract contains a non-scalar monitoring value")


def _error_response(status_code: int, detail: str, request_id: UUID) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail, "request_id": str(request_id)},
        headers={"X-Request-ID": str(request_id), "X-Content-Type-Options": "nosniff"},
    )
