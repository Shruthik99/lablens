.PHONY: setup lint test build-lexicon eval eval-track eval-llm eval-t5 mlflow-ui api up down logs

setup:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev,api]"
	python -m pre_commit install

lint:
	python -m ruff check .
	python -m black --check .

test:
	python -m pytest

build-lexicon:
	python -m lablens.rxnorm

eval:
	python -m lablens.evaluate tests/fixtures/mini_eval.json

eval-track:
	python -m lablens.evaluate tests/fixtures/mini_eval.json --track

eval-llm:
	python -m lablens.evaluate tests/fixtures/mini_eval.json --extractor llm --track

eval-t5:
	python -m lablens.evaluate tests/fixtures/mini_eval.json --extractor t5 --track

mlflow-ui:
	python -m mlflow ui --backend-store-uri sqlite:///mlflow.db

api:
	python -m uvicorn lablens.api:app --reload --port 8000

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f
