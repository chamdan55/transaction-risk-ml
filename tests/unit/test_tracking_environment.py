import pytest

from pipelines.validate_tracking_env import (
    TrackingEnvironmentError,
    validate_tracking_environment,
)


def test_tracking_environment_reads_required_values_from_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "POSTGRES_PASSWORD=0123456789abcdef0123456789abcdef\n"
        "MINIO_ROOT_USER=0123456789abcdef\n"
        "MINIO_ROOT_PASSWORD=abcdef0123456789abcdef0123456789\n"
        "MLFLOW_PORT=5050\n",
        encoding="utf-8",
    )

    validate_tracking_environment(env_file=env_file, process_env={})


def test_tracking_environment_reports_missing_names_without_values(tmp_path):
    with pytest.raises(TrackingEnvironmentError, match="POSTGRES_PASSWORD.*MINIO_ROOT_USER"):
        validate_tracking_environment(env_file=tmp_path / "missing.env", process_env={})


def test_tracking_environment_rejects_unsafe_database_password(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "POSTGRES_PASSWORD=password-with-symbols!\n"
        "MINIO_ROOT_USER=local-user\n"
        "MINIO_ROOT_PASSWORD=long-enough-password\n",
        encoding="utf-8",
    )

    with pytest.raises(TrackingEnvironmentError, match="hexadecimal"):
        validate_tracking_environment(env_file=env_file, process_env={})


@pytest.mark.parametrize("port", ("0", "65536", "not-a-number"))
def test_tracking_environment_rejects_invalid_port(tmp_path, port):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "POSTGRES_PASSWORD=0123456789abcdef0123456789abcdef\n"
        "MINIO_ROOT_USER=local-user\n"
        "MINIO_ROOT_PASSWORD=long-enough-password\n",
        encoding="utf-8",
    )

    with pytest.raises(TrackingEnvironmentError, match="MLFLOW_PORT"):
        validate_tracking_environment(
            env_file=env_file,
            process_env={"MLFLOW_PORT": port},
        )
