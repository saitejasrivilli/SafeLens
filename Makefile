.PHONY: install test lint typecheck format ci data-download data-validate data-prepare data-test

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

data-download:
	.venv/bin/python scripts/download_data.py

data-validate:
	.venv/bin/python scripts/validate_data.py

data-prepare:
	.venv/bin/python scripts/prepare_data.py

data-test:
	.venv/bin/pytest -v tests/unit/data
