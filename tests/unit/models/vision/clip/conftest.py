"""Uses a tiny random CLIP checkpoint (~1M params) instead of the real
openai/clip-vit-base-patch32 (~151M params) so unit tests stay fast and
don't require downloading the production model."""

import pytest
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel

TINY_CLIP_MODEL = "hf-internal-testing/tiny-random-CLIPModel"


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


@pytest.fixture
def sample_image() -> Image.Image:
    return Image.new("RGB", (64, 64), color=(10, 20, 30))
