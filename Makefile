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

baseline-train:
	.venv/bin/python scripts/train_baseline.py

baseline-test:
	.venv/bin/pytest -v tests/unit/models

deberta-train:
	.venv/bin/python scripts/train_deberta.py

deberta-test:
	.venv/bin/pytest -v tests/unit/models/deberta

multimodal-download:
	.venv/bin/python scripts/download_multimodal_data.py

multimodal-validate:
	.venv/bin/python scripts/validate_multimodal_data.py

multimodal-remediate:
	.venv/bin/python scripts/remediate_multimodal_leakage.py

multimodal-test:
	.venv/bin/pytest -v tests/unit/data/multimodal

image-baseline-train:
	.venv/bin/python scripts/train_image_baseline.py

image-baseline-test:
	.venv/bin/pytest -v tests/unit/models/vision

text-arabic-train-arabert:
	.venv/bin/python scripts/train_text_arabic_baseline.py --config configs/text_arabic_arabert.yaml

text-arabic-train-mdeberta:
	.venv/bin/python scripts/train_text_arabic_baseline.py --config configs/text_arabic_mdeberta.yaml

text-arabic-test:
	.venv/bin/pytest -v tests/unit/models/text_multilingual

multimodal-fusion-train:
	.venv/bin/python scripts/train_multimodal_fusion.py

multimodal-fusion-test:
	.venv/bin/pytest -v tests/unit/models/multimodal
