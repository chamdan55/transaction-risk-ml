CONTAINER_ENGINE ?= podman
COMPOSE ?= $(CONTAINER_ENGINE) compose

.PHONY: install test lint-fix lint format format-check pre-commit pipeline train run up-mlflow up up-tracking down down-clean

install:
	python -m pip install --upgrade pip
	pip install -e ".[training,tracking,dev]" -c requirements/constraints-py312.txt
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

run:
	uvicorn app.main:app --reload

up-mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root mlartifacts

up:
	$(COMPOSE) up --build -d

up-tracking:
	$(COMPOSE) --profile tracking up --build -d

down:
	$(COMPOSE) down --remove-orphans

down-clean:
	$(COMPOSE) down --volumes --remove-orphans
