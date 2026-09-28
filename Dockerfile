# The default is a patch-level Python tag. CI or a dependency bot may pass a
# digest-pinned value, for example: --build-arg PYTHON_IMAGE=python:3.12.14-slim-bookworm@sha256:...
ARG PYTHON_IMAGE=python:3.12.14-slim-bookworm

FROM ${PYTHON_IMAGE} AS wheel-builder-base

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY pyproject.toml README.md ./
COPY requirements/constraints-py312.txt requirements/constraints-py312.txt
COPY app/ app/
COPY ml/ ml/
COPY pipelines/ pipelines/

FROM wheel-builder-base AS serving-wheel-builder

RUN python -m pip install --upgrade pip \
    && python -m pip wheel --wheel-dir /wheels/serving \
        ".[serving]" -c requirements/constraints-py312.txt

FROM wheel-builder-base AS training-wheel-builder

RUN python -m pip install --upgrade pip \
    && python -m pip wheel --wheel-dir /wheels/training \
        ".[training,tracking]" -c requirements/constraints-py312.txt

FROM wheel-builder-base AS tracking-wheel-builder

RUN python -m pip install --upgrade pip \
    && python -m pip wheel --wheel-dir /wheels/tracking \
        ".[tracking]" -c requirements/constraints-py312.txt

FROM wheel-builder-base AS monitoring-wheel-builder

RUN python -m pip install --upgrade pip \
    && python -m pip wheel --wheel-dir /wheels/monitoring \
        ".[monitoring]" -c requirements/constraints-py312.txt

FROM ${PYTHON_IMAGE} AS runtime-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN addgroup --system --gid 10001 app \
    && adduser --system --uid 10001 --ingroup app --home /app app

RUN mkdir -p /var/lib/transaction-risk/monitoring \
    && chown -R app:app /var/lib/transaction-risk

WORKDIR /app

FROM runtime-base AS serving

COPY --from=serving-wheel-builder /wheels/serving /wheels
RUN python -m pip install --no-index --find-links=/wheels transaction-risk-ml[serving] \
    && rm -rf /wheels

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "from urllib.request import urlopen; assert urlopen('http://127.0.0.1:8000/health/live', timeout=2).status == 200"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]

FROM runtime-base AS training

RUN apt-get update \
    && apt-get install --no-install-recommends --yes openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

COPY --from=training-wheel-builder /wheels/training /wheels
COPY --chown=app:app configs/ /workspace/configs/
RUN python -m pip install --no-index --find-links=/wheels transaction-risk-ml[training,tracking] \
    && rm -rf /wheels \
    && mkdir -p /workspace/artifacts \
    && chown -R app:app /workspace

USER app
WORKDIR /workspace

CMD ["python", "-m", "pipelines.train_models"]

FROM runtime-base AS tracking-server

COPY --from=tracking-wheel-builder /wheels/tracking /wheels
RUN python -m pip install --no-index --find-links=/wheels transaction-risk-ml[tracking] \
    && rm -rf /wheels

USER app
WORKDIR /workspace

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]

FROM runtime-base AS monitoring

COPY --from=monitoring-wheel-builder /wheels/monitoring /wheels
RUN python -m pip install --no-index --find-links=/wheels transaction-risk-ml[monitoring] \
    && rm -rf /wheels

USER app
WORKDIR /workspace

CMD ["python", "-m", "pipelines.run_monitoring"]

FROM busybox:1.37.0-musl AS model-bundle

ARG MODEL_BUNDLE_SOURCE=artifacts
COPY ${MODEL_BUNDLE_SOURCE}/models/ /bundle/
COPY ${MODEL_BUNDLE_SOURCE}/evaluation_report.json /bundle/evaluation_report.json

FROM quay.io/minio/mc:RELEASE.2025-02-08T19-14-21Z AS minio-mc

FROM busybox:1.37.0-musl AS minio-init

COPY --from=minio-mc /usr/bin/mc /usr/bin/mc

# Compose needs a self-contained API image. Keeping this as a separate target
# lets CI build and scan the generic serving image without local model artifacts.
FROM serving AS serving-with-model

COPY --from=model-bundle --chown=app:app /bundle/ /models/
