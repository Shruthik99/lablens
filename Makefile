.PHONY: setup lint test eval

setup:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"
	python -m pre_commit install

lint:
	python -m ruff check .
	python -m black --check .

test:
	python -m pytest

eval:
	python -m lablens.evaluate tests/fixtures/mini_eval.json
