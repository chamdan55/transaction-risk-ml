from functools import lru_cache
from pathlib import Path
from typing import Literal

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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
