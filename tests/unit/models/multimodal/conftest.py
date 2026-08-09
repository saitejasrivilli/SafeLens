"""Tiny random CLIP + BERT checkpoints -- not the real
openai/clip-vit-base-patch32 / aubmindlab/bert-base-arabertv2 -- so unit
tests stay fast and don't require downloading the production models."""

import pytest
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, CLIPModel

TINY_CLIP_MODEL = "hf-internal-testing/tiny-random-CLIPModel"
TINY_TEXT_MODEL = "hf-internal-testing/tiny-random-bert"


@pytest.fixture(scope="session")
def tiny_clip_processor():
    return CLIPImageProcessor.from_pretrained(TINY_CLIP_MODEL)


@pytest.fixture(scope="session")
def tiny_clip_model():
    model = CLIPModel.from_pretrained(TINY_CLIP_MODEL)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@pytest.fixture(scope="session")
def tiny_text_tokenizer():
    return AutoTokenizer.from_pretrained(TINY_TEXT_MODEL)


@pytest.fixture(scope="session")
def tiny_text_model():
    model = AutoModel.from_pretrained(TINY_TEXT_MODEL)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@pytest.fixture
def sample_image() -> Image.Image:
    return Image.new("RGB", (64, 64), color=(10, 20, 30))
