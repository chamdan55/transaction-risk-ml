.PHONY: install test lint-fix lint format format-check pre-commit run

install:
	python -m pip install --upgrade pip
	pip install -e ".[dev]"

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

run:
	uvicorn app.main:app --reload
