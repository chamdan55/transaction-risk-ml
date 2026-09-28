from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_container_definition_separates_serving_and_training_dependencies() -> None:
    dockerfile = (REPOSITORY_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM runtime-base AS serving" in dockerfile
    assert "transaction-risk-ml[serving]" in dockerfile
    assert "FROM runtime-base AS training" in dockerfile
    assert "transaction-risk-ml[training,tracking]" in dockerfile
    assert "pyspark" not in dockerfile.split("FROM runtime-base AS training")[0]


def test_compose_keeps_stateful_services_private_and_uses_a_bundled_model() -> None:
    compose = (REPOSITORY_ROOT / "compose.yaml").read_text(encoding="utf-8")
    dockerfile = (REPOSITORY_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "127.0.0.1:${API_PORT:-8000}:8000" in compose
    assert "target: serving-with-model" in compose
    assert "model-seed:" not in compose
    assert "model-artifacts:/models" not in compose
    assert "FROM serving AS serving-with-model" in dockerfile
    assert "COPY --from=model-bundle --chown=app:app /bundle/ /models/" in dockerfile
    assert "postgres-data:/var/lib/postgresql/data" in compose
    assert "--backend-store-uri postgresql+psycopg2://" in compose
    assert "psycopg2-binary" in (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "minio-data:/data" in compose
    assert "internal: true" in compose
    assert "postgres:\n    image" in compose
    assert "minio:\n    image" in compose
