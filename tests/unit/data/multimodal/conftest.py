import pytest

from safelens.data.multimodal.schema import MultimodalExample


def make_example(
    example_id: str, text: str, image_path: str, hate_label: int = 0, **overrides
) -> MultimodalExample:
    defaults = dict(
        example_id=example_id,
        text=text,
        image_path=image_path,
        source_img_path=f"./{image_path}",
        hate_label=hate_label,
        prop_label=0,
        hate_fine_grained_label=1,
        split="train",
        dataset_version="QCRI/Prop2Hate-Meme@test-fixture",
    )
    defaults.update(overrides)
    return MultimodalExample(**defaults)


@pytest.fixture
def make_example_fn():
    return make_example


@pytest.fixture
def tiny_jpeg(tmp_path):
    """Writes a real, tiny, valid JPEG file and returns its path."""
    from PIL import Image

    path = tmp_path / "tiny.jpg"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(path, format="JPEG")
    return path
