"""Uses tiny random BERT/DeBERTa-v2 checkpoints instead of the real
AraBERT/mDeBERTa (~135M/278M params) so unit tests stay fast and don't
require downloading the production models."""

import pytest
from transformers import AutoModel, AutoTokenizer

TINY_BERT = "hf-internal-testing/tiny-random-bert"
TINY_DEBERTA = "hf-internal-testing/tiny-random-DebertaV2Model"


@pytest.fixture(scope="session")
def tiny_text_tokenizer():
    return AutoTokenizer.from_pretrained(TINY_BERT)


@pytest.fixture(scope="session")
def tiny_text_model():
    model = AutoModel.from_pretrained(TINY_BERT)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model
