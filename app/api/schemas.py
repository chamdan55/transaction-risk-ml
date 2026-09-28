"""Strict public schemas for synchronous pre-transaction scoring."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, field_validator

from ml.monitoring.contracts import DELAYED_LABEL_SCHEMA_VERSION


class TransactionType(StrEnum):
    """Transaction categories supported by the current feature contract."""

    CASH_IN = "CASH_IN"
    CASH_OUT = "CASH_OUT"
    DEBIT = "DEBIT"
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"


class PredictionRequest(BaseModel):
    """Only fields available before transaction settlement are accepted."""

    # JSON represents timestamps, UUIDs, and enums as strings.  Keep those public
    # representations valid while requiring numeric transaction values to be numeric.
    model_config = ConfigDict(extra="forbid")

    transaction_type: TransactionType
    amount: StrictFloat = Field(ge=0)
    origin_balance_before: StrictFloat = Field(ge=0)
    destination_balance_before: StrictFloat = Field(ge=0)
    timestamp: datetime
    request_id: UUID | None = None


class PredictionResponse(BaseModel):
    """Stable, auditable response contract for a single prediction."""

    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    feedback_id: UUID
    risk_score: float = Field(ge=0, le=1)
    decision: str
    model_name: str
    model_version: str
    model_source: str
    threshold: float = Field(ge=0, le=1)
    threshold_policy: str
    feature_contract_version: str


class DelayedLabelRequest(BaseModel):
    """Ground truth sent later using the pseudonymous ID returned by inference."""

    model_config = ConfigDict(extra="forbid")

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


class DelayedLabelResponse(BaseModel):
    schema_version: Literal["delayed-label-v1"] = DELAYED_LABEL_SCHEMA_VERSION
    feedback_id: UUID
    status: Literal["accepted", "duplicate"]


class HealthResponse(BaseModel):
    status: str
    reason: str | None = None


class ModelInfoResponse(BaseModel):
    status: str
    model_name: str | None = None
    model_version: str | None = None
    model_source: str | None = None
    threshold: float | None = None
    threshold_policy: str | None = None
    feature_contract_version: str | None = None
    reason: str | None = None
