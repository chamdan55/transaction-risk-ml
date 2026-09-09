from app.core.config import Settings


def test_default_settings():
    settings = Settings()

    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.model_name == "transaction-risk-model"
    assert settings.model_version == "local"
