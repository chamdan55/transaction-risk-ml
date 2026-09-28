CONTAINER_ENGINE ?= podman
COMPOSE ?= $(CONTAINER_ENGINE) compose

.PHONY: install test lint-fix lint format format-check pre-commit pipeline train train-tracking prepare-retraining-manifest retrain-tracking check-tracking check-tracking-env tracking-smoke promote-tracking rollback-tracking export-production-model kind-rollout run up-mlflow up up-tracking up-monitoring monitoring-report monitoring-report-compose down down-clean load kind-up kind-down kind-apply kind-rollback

MLFLOW_PORT ?= 5000
APPROVED_BY ?= Camdun
APPROVAL_REASON ?= Better_than_the_previous_model
APPROVED_MANIFEST ?= artifacts/retraining/approved-manifest.json
RETRAINING_POLICY ?= configs/retraining.yaml
MODEL_VERSION ?= 2
PROMOTION_STAGE ?= production
PROMOTION_REASON ?= Better_than_the_previous_model
ROLLBACK_STAGE ?= production
ROLLBACK_REASON ?=
MODEL_BUNDLE_PATH ?=

install:
	python -m pip install --upgrade pip
	pip install -e ".[training,tracking,serving,monitoring,dev]" -c requirements/constraints-py312.txt
	pre-commit install

test:
	pytest

lint-fix:
	ruff check . --fix

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

pre-commit:
	pre-commit run --all-files

pipeline:
	python -m pipelines.run_pipeline

train:
	python -m pipelines.train_models

train-tracking:
	python -m pipelines.train_models \
		--tracking-config configs/tracking-compose.yaml \
		--tracking-uri http://127.0.0.1:$(MLFLOW_PORT)

prepare-retraining-manifest:
	python -m pipelines.prepare_retraining_manifest \
		--config configs/model.yaml \
		--output "$(APPROVED_MANIFEST)" \
		--approved-by "$(APPROVED_BY)" \
		--reason "$(APPROVAL_REASON)"

retrain-tracking:
	python -m pipelines.retrain_model \
		--config configs/model.yaml \
		--policy "$(RETRAINING_POLICY)" \
		--approved-manifest "$(APPROVED_MANIFEST)" \
		--tracking-config configs/tracking-compose.yaml \
		--tracking-uri http://127.0.0.1:$(MLFLOW_PORT)

check-tracking:
	python -m pipelines.check_tracking \
		--tracking-config configs/tracking-compose.yaml \
		--tracking-uri http://127.0.0.1:$(MLFLOW_PORT)

tracking-smoke:
	python -m pipelines.smoke_mlflow_tracking \
		--tracking-config configs/tracking-compose.yaml \
		--tracking-uri http://127.0.0.1:$(MLFLOW_PORT)

promote-tracking:
	python -m pipelines.promote_model \
		--tracking-config configs/tracking-compose.yaml \
		--tracking-uri http://127.0.0.1:$(MLFLOW_PORT) \
		--version "$(MODEL_VERSION)" \
		--stage "$(PROMOTION_STAGE)" \
		--approved-by "$(APPROVED_BY)" \
		--reason "$(PROMOTION_REASON)"

rollback-tracking:
	python -m pipelines.rollback_model \
		--tracking-config configs/tracking-compose.yaml \
		--tracking-uri http://127.0.0.1:$(MLFLOW_PORT) \
		--stage "$(ROLLBACK_STAGE)" \
		--approved-by "$(APPROVED_BY)" \
		--reason "$(ROLLBACK_REASON)"

export-production-model:
	python -m pipelines.export_model_bundle \
		--tracking-config configs/tracking-compose.yaml \
		--tracking-uri http://127.0.0.1:$(MLFLOW_PORT)

kind-rollout:
	python -m pipelines.rollout_model \
		--bundle-path "$(MODEL_BUNDLE_PATH)" \
		--tracking-config configs/tracking-compose.yaml \
		--tracking-uri http://127.0.0.1:$(MLFLOW_PORT)

run:
	uvicorn app.main:app --reload

up-mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root mlartifacts

up-build:
	python -m pipelines.validate_tracking_env
	$(COMPOSE) --profile monitoring --profile tracking up --build -d

up:
	python -m pipelines.validate_tracking_env
	$(COMPOSE) --profile monitoring --profile tracking up -d

up-tracking:
	python -m pipelines.validate_tracking_env
	$(COMPOSE) --profile tracking up --build -d postgres minio minio-init mlflow

check-tracking-env:
	python -m pipelines.validate_tracking_env

up-monitoring:
	$(COMPOSE) --profile monitoring up --build -d

monitoring-report:
	python -m pipelines.run_monitoring

monitoring-report-compose:
	$(COMPOSE) --profile monitoring-job build monitoring-report
	$(COMPOSE) --profile monitoring-job run --rm monitoring-report

down:
	$(COMPOSE) --profile monitoring --profile tracking down --remove-orphans

down-clean:
	$(COMPOSE) --profile monitoring --profile tracking down --volumes --remove-orphans

# Requires k6. Set BASE_URL and K6_API_KEY in the environment when needed.
load:
	k6 run tests/load/transaction_risk.js

# kind uses Podman when KIND_EXPERIMENTAL_PROVIDER=podman is set in the shell.
kind-up:
	kind create cluster --name transaction-risk --config deployment/kubernetes/kind-config.yaml

kind-apply:
	kubectl apply -k deployment/kubernetes
	kubectl -n transaction-risk rollout status deployment/transaction-risk-api --timeout=3m

kind-rollback:
	kubectl -n transaction-risk rollout undo deployment/transaction-risk-api
	kubectl -n transaction-risk rollout status deployment/transaction-risk-api --timeout=3m

kind-down:
	kind delete cluster --name transaction-risk
