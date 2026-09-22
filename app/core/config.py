from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    environment: str = "development"
    log_level: str = "INFO"

    model_name: str = "transaction-risk-model"
    model_version: str = "local"
    model_source: Literal["local", "mlflow_alias"] = "local"
    model_artifact_path: Path = Path("artifacts/models/random_forest.joblib")
    model_evaluation_report_path: Path = Path("artifacts/evaluation_report.json")
    model_registry_alias: str = "production"

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"

    api_auth_enabled: bool = False
    api_key: SecretStr | None = None
    max_request_bytes: int = Field(default=65_536, ge=1_024, le=1_048_576)
    max_concurrent_predictions: int = Field(default=8, ge=1, le=128)
    prediction_queue_timeout_seconds: float = Field(default=0.05, gt=0, le=5)
    prediction_timeout_seconds: float = Field(default=2.0, gt=0, le=30)
    shutdown_grace_seconds: float = Field(default=5.0, gt=0, le=30)
    idempotency_ttl_seconds: float = Field(default=300.0, gt=0, le=3_600)
    idempotency_max_entries: int = Field(default=1_000, ge=1, le=100_000)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_authentication_settings(self) -> "Settings":
        if self.api_auth_enabled and (
            self.api_key is None or not self.api_key.get_secret_value().strip()
        ):
            raise ValueError("api_key must be configured when api_auth_enabled is true")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
