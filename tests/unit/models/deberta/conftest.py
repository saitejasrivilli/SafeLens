"""Uses a tiny random DeBERTa-v2 checkpoint (~85K params) instead of the real
microsoft/deberta-v3-small (~142M params) so unit tests stay fast and don't
require downloading the production model."""

import pytest
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TINY_MODEL = "ydshieh/tiny-random-DebertaV2ForSequenceClassification"


@pytest.fixture(scope="session")
def tiny_tokenizer():
    return AutoTokenizer.from_pretrained(TINY_MODEL)


@pytest.fixture(scope="session")
def tiny_model():
    return AutoModelForSequenceClassification.from_pretrained(TINY_MODEL, num_labels=2)
