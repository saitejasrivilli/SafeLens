from pathlib import Path

import torch
from PIL import Image

from safelens.models.multimodal.dataset import extract_fused_embeddings


class _Example:
    def __init__(self, example_id: str, text: str, image_path: str, hate_label: int):
        self.example_id = example_id
        self.text = text
        self.image_path = image_path
        self.hate_label = hate_label


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path, format="JPEG")


def test_fused_embeddings_shape_and_concat_order(
    tmp_path: Path, tiny_clip_processor, tiny_clip_model, tiny_text_tokenizer, tiny_text_model
):
    _write_image(tmp_path / "a.jpg", (10, 20, 30))
    _write_image(tmp_path / "b.jpg", (40, 50, 60))
    examples = [
        _Example("id1", "some text one", "a.jpg", 0),
        _Example("id2", "some text two", "b.jpg", 1),
    ]

    result = extract_fused_embeddings(
        examples,
        tmp_path,
        tiny_clip_processor,
        tiny_clip_model,
        tiny_text_tokenizer,
        tiny_text_model,
        "cpu",
        max_seq_length=16,
        batch_size=2,
    )

    image_dim = tiny_clip_model.config.projection_dim
    text_dim = tiny_text_model.config.hidden_size

    assert result.image_embeddings.shape == (2, image_dim)
    assert result.text_embeddings.shape == (2, text_dim)
    assert result.fused_embeddings.shape == (2, text_dim + image_dim)
    assert result.labels.tolist() == [0, 1]
    assert result.example_ids == ["id1", "id2"]

    # Concatenation order is [text ; image] -- verify by slicing.
    torch.testing.assert_close(result.fused_embeddings[:, :text_dim], result.text_embeddings)
    torch.testing.assert_close(result.fused_embeddings[:, text_dim:], result.image_embeddings)


def test_fused_embeddings_deterministic(
    tmp_path: Path, tiny_clip_processor, tiny_clip_model, tiny_text_tokenizer, tiny_text_model
):
    _write_image(tmp_path / "a.jpg", (5, 5, 5))
    examples = [_Example("id1", "deterministic text", "a.jpg", 0)]

    result_a = extract_fused_embeddings(
        examples,
        tmp_path,
        tiny_clip_processor,
        tiny_clip_model,
        tiny_text_tokenizer,
        tiny_text_model,
        "cpu",
        16,
    )
    result_b = extract_fused_embeddings(
        examples,
        tmp_path,
        tiny_clip_processor,
        tiny_clip_model,
        tiny_text_tokenizer,
        tiny_text_model,
        "cpu",
        16,
    )

    torch.testing.assert_close(result_a.fused_embeddings, result_b.fused_embeddings)
