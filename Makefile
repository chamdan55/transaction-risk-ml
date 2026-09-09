.PHONY: install test lint format format-check run

install:
	python -m pip install --upgrade pip
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

run:
	uvicorn app.main:app --reload
