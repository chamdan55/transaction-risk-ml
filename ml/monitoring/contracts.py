"""Versioned, privacy-conscious contracts for prediction monitoring events."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
)

from ml.contracts.features import MODEL_FEATURE_COLUMNS

PREDICTION_EVENT_SCHEMA_VERSION = "prediction-event-v1"
DELAYED_LABEL_SCHEMA_VERSION = "delayed-label-v1"
_ALLOWED_TRANSACTION_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}


class PredictionEvent(BaseModel):
    """One successful prediction with only contract-approved monitoring features."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["prediction-event-v1"] = PREDICTION_EVENT_SCHEMA_VERSION
    feedback_id: UUID
    event_time: datetime
    model_name: str = Field(min_length=1, max_length=128)
    model_version: str = Field(min_length=1, max_length=128)
    feature_contract_version: str = Field(min_length=1, max_length=128)
    features: dict[str, StrictStr | StrictInt | StrictFloat]
    risk_score: float = Field(ge=0, le=1)
    decision_threshold: float = Field(ge=0, le=1)
    decision: Literal["allow", "review"]
    latency_seconds: float = Field(ge=0)

    @field_validator("event_time")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event_time must include a timezone")
        return value

    @field_validator("features")
    @classmethod
    def validate_approved_features(cls, value: dict[str, StrictStr | StrictInt | StrictFloat]):
        if set(value) != set(MODEL_FEATURE_COLUMNS):
            raise ValueError("features must exactly match the approved model feature contract")
        for name, feature_value in value.items():
            if isinstance(feature_value, bool) or not isinstance(feature_value, (str, int, float)):
                raise ValueError("monitoring features must be scalar contract values")
            if isinstance(feature_value, str) and (
                name != "transaction_type" or feature_value not in _ALLOWED_TRANSACTION_TYPES
            ):
                raise ValueError("monitoring string feature is not an approved transaction type")
            if isinstance(feature_value, float) and not isfinite(feature_value):
                raise ValueError("monitoring features must be finite")
        return value


class DelayedLabel(BaseModel):
    """A single final binary outcome for a previously returned feedback ID."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["delayed-label-v1"]
    feedback_id: UUID
    label: StrictInt = Field(ge=0, le=1)
    label_time: datetime

    @field_validator("label_time")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("label_time must include a timezone")
        if value.astimezone(UTC) > datetime.now(UTC) + timedelta(minutes=5):
            raise ValueError("label_time cannot be more than five minutes in the future")
        return value
