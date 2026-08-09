.PHONY: install test lint typecheck format ci

install:
	uv venv --python 3.11 .venv
	uv pip install -e ".[dev]" --python .venv/bin/python

test:
	.venv/bin/pytest -v

lint:
	.venv/bin/ruff check .

format:
	.venv/bin/ruff format .

typecheck:
	.venv/bin/mypy src

ci: lint typecheck test
