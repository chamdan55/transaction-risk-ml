"""Shared conversion from a pre-transaction request to the model feature frame."""

from __future__ import annotations

from math import log1p
from typing import Any

import pandas as pd

from app.api.schemas import PredictionRequest, TransactionType
from ml.contracts.features import MODEL_FEATURE_COLUMNS, validate_model_feature_columns


def prepare_feature_frame(request: PredictionRequest) -> pd.DataFrame:
    """Build the exact ordered training feature contract from safe request inputs.

    The derivations intentionally mirror the Spark feature pipeline.  In particular,
    Spark's ``dayofweek`` uses Sunday=1 through Saturday=7.
    """

    transaction_type = request.transaction_type.value
    timestamp = request.timestamp
    amount = request.amount
    origin_balance = request.origin_balance_before
    destination_balance = request.destination_balance_before
    values: dict[str, Any] = {
        "transaction_type": transaction_type,
        "amount": amount,
        "amount_log": log1p(amount),
        "origin_balance_before": origin_balance,
        "amount_to_origin_balance_ratio": amount / origin_balance if origin_balance > 0 else 0.0,
        "destination_balance_before": destination_balance,
        "amount_to_destination_balance_ratio": (
            amount / destination_balance if destination_balance > 0 else 0.0
        ),
        "is_zero_origin_balance_before": int(origin_balance == 0),
        "is_zero_destination_balance_before": int(destination_balance == 0),
        "transaction_hour": timestamp.hour,
        "transaction_day_of_week": (timestamp.weekday() + 1) % 7 + 1,
        "is_cash_in": int(transaction_type == TransactionType.CASH_IN.value),
        "is_cash_out": int(transaction_type == TransactionType.CASH_OUT.value),
        "is_debit": int(transaction_type == TransactionType.DEBIT.value),
        "is_payment": int(transaction_type == TransactionType.PAYMENT.value),
        "is_transfer": int(transaction_type == TransactionType.TRANSFER.value),
    }
    feature_columns = validate_model_feature_columns(values)
    if feature_columns != MODEL_FEATURE_COLUMNS:  # pragma: no cover - contract invariant
        raise RuntimeError("Serving feature contract is inconsistent")
    return pd.DataFrame([values], columns=feature_columns)
