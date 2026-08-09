from pathlib import Path

import torch
from PIL import Image

from safelens.models.vision.clip.dataset import extract_embeddings

EMBED_DIM = 64


class _NoTextExample:
    """Duck-typed stand-in for MultimodalExample whose `text` attribute
    raises if ever accessed -- proves extract_embeddings truly never
    touches the text field."""

    def __init__(self, example_id: str, image_path: str, hate_label: int):
        self.example_id = example_id
        self.image_path = image_path
        self.hate_label = hate_label

    @property
    def text(self):
        raise AssertionError("image-only pipeline must never access .text")


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path, format="JPEG")


def test_extract_embeddings_never_touches_text(
    tmp_path: Path, tiny_clip_processor, tiny_clip_model
):
    _write_image(tmp_path / "a.jpg", (10, 20, 30))
    _write_image(tmp_path / "b.jpg", (40, 50, 60))
    examples = [
        _NoTextExample("id1", "a.jpg", 0),
        _NoTextExample("id2", "b.jpg", 1),
    ]

    embeddings, labels, ids = extract_embeddings(
        examples, tmp_path, tiny_clip_processor, tiny_clip_model, "cpu", batch_size=2
    )

    assert embeddings.shape == (2, tiny_clip_model.config.projection_dim)
    assert labels.tolist() == [0, 1]
    assert ids == ["id1", "id2"]


def test_extract_embeddings_batching_matches_single(
    tmp_path: Path, tiny_clip_processor, tiny_clip_model
):
    _write_image(tmp_path / "a.jpg", (10, 20, 30))
    _write_image(tmp_path / "b.jpg", (40, 50, 60))
    examples = [
        _NoTextExample("id1", "a.jpg", 0),
        _NoTextExample("id2", "b.jpg", 1),
    ]

    batched, _, _ = extract_embeddings(
        examples, tmp_path, tiny_clip_processor, tiny_clip_model, "cpu", batch_size=2
    )
    one_at_a_time, _, _ = extract_embeddings(
        examples, tmp_path, tiny_clip_processor, tiny_clip_model, "cpu", batch_size=1
    )

    torch.testing.assert_close(batched, one_at_a_time)
