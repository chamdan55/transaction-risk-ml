"""Versioned prediction-time feature contract.

The complete Spark feature schema intentionally contains canonical and audit fields that are useful
for profiling.  This module defines the smaller, ordered feature set that is allowed to reach the
pre-transaction risk model.  Training and serving code must import this contract instead of
deriving model columns from the complete feature schema.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

FEATURE_CONTRACT_VERSION = "pre-transaction-v1"


@dataclass(frozen=True)
class FeatureDefinition:
    """Metadata for one model feature at prediction time."""

    name: str
    availability: str
    description: str


# These fields are available before transaction settlement and can be represented in a synchronous
# request. Historical/velocity features are deliberately deferred until an online state source is
# implemented; serving must never silently invent their values.
FEATURE_DEFINITIONS = (
    FeatureDefinition("transaction_type", "request", "Transaction type/category."),
    FeatureDefinition("amount", "request", "Transaction amount."),
    FeatureDefinition("amount_log", "derived", "log1p(amount)."),
    FeatureDefinition("origin_balance_before", "request", "Origin balance before transaction."),
    FeatureDefinition(
        "amount_to_origin_balance_ratio",
        "derived",
        "Amount divided by origin balance before transaction.",
    ),
    FeatureDefinition(
        "destination_balance_before",
        "request",
        "Destination balance before transaction.",
    ),
    FeatureDefinition(
        "amount_to_destination_balance_ratio",
        "derived",
        "Amount divided by destination balance before transaction.",
    ),
    FeatureDefinition(
        "is_zero_origin_balance_before",
        "derived",
        "Whether the origin balance is zero before the transaction.",
    ),
    FeatureDefinition(
        "is_zero_destination_balance_before",
        "derived",
        "Whether the destination balance is zero before the transaction.",
    ),
    FeatureDefinition("transaction_hour", "derived", "Hour derived from event timestamp."),
    FeatureDefinition(
        "transaction_day_of_week",
        "derived",
        "Day of week derived from event timestamp.",
    ),
    FeatureDefinition("is_cash_in", "derived", "Transaction type indicator."),
    FeatureDefinition("is_cash_out", "derived", "Transaction type indicator."),
    FeatureDefinition("is_debit", "derived", "Transaction type indicator."),
    FeatureDefinition("is_payment", "derived", "Transaction type indicator."),
    FeatureDefinition("is_transfer", "derived", "Transaction type indicator."),
)

MODEL_FEATURE_COLUMNS = tuple(definition.name for definition in FEATURE_DEFINITIONS)

POST_EVENT_FEATURE_COLUMNS = frozenset(
    {
        "origin_balance_after",
        "destination_balance_after",
        "origin_balance_delta",
        "destination_balance_delta",
        "origin_balance_change_ratio",
        "destination_balance_change_ratio",
        "origin_balance_mismatch",
        "destination_balance_mismatch",
        "origin_balance_depleted",
        "destination_balance_increased",
    }
)

HISTORICAL_FEATURE_COLUMNS = frozenset(
    {
        "transactions_last_1h",
        "transactions_last_24h",
        "amount_sum_last_24h",
        "unique_destinations_last_30d",
    }
)

FORBIDDEN_MODEL_FEATURE_COLUMNS = frozenset(
    {
        "transaction_id",
        "timestamp",
        "origin_account_id",
        "destination_account_id",
        "is_fraud",
        "is_flagged_fraud",
        *POST_EVENT_FEATURE_COLUMNS,
        *HISTORICAL_FEATURE_COLUMNS,
    }
)


def validate_model_feature_columns(columns: Iterable[str]) -> tuple[str, ...]:
    """Validate and normalize an ordered model feature list."""

    normalized = tuple(columns)
    if normalized != MODEL_FEATURE_COLUMNS:
        raise ValueError(
            "Model features must exactly match the ordered "
            f"{FEATURE_CONTRACT_VERSION} contract: {MODEL_FEATURE_COLUMNS}"
        )
    forbidden = FORBIDDEN_MODEL_FEATURE_COLUMNS.intersection(normalized)
    if forbidden:
        raise ValueError(f"Forbidden model features: {sorted(forbidden)}")
    return normalized
