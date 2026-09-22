"""Shared contracts used by offline training and online inference."""

from ml.contracts.features import (
    FEATURE_CONTRACT_VERSION,
    FORBIDDEN_MODEL_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    validate_model_feature_columns,
)

__all__ = [
    "FEATURE_CONTRACT_VERSION",
    "FORBIDDEN_MODEL_FEATURE_COLUMNS",
    "MODEL_FEATURE_COLUMNS",
    "validate_model_feature_columns",
]
