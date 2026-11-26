.PHONY: setup lint test eval eval-track mlflow-ui api

setup:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev,api]"
	python -m pre_commit install

lint:
	python -m ruff check .
	python -m black --check .

test:
	python -m pytest

eval:
	python -m lablens.evaluate tests/fixtures/mini_eval.json

eval-track:
	python -m lablens.evaluate tests/fixtures/mini_eval.json --track

mlflow-ui:
	python -m mlflow ui --backend-store-uri sqlite:///mlflow.db

api:
	python -m uvicorn lablens.api:app --reload --port 8000
