import torch

from safelens.models.multimodal.missing_modality import (
    build_image_only_input,
    build_text_only_input,
    compute_modality_means,
)


def test_compute_modality_means_shapes():
    image_emb = torch.randn(10, 4)
    text_emb = torch.randn(10, 6)
    means = compute_modality_means(image_emb, text_emb)
    assert means.image_mean.shape == (4,)
    assert means.text_mean.shape == (6,)


def test_compute_modality_means_values():
    image_emb = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    text_emb = torch.tensor([[10.0], [20.0]])
    means = compute_modality_means(image_emb, text_emb)
    torch.testing.assert_close(means.image_mean, torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(means.text_mean, torch.tensor([15.0]))


def test_build_image_only_input_replaces_text_with_mean():
    image_emb = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    text_emb = torch.tensor([[10.0], [20.0]])
    means = compute_modality_means(image_emb, text_emb)

    fused_image_only = build_image_only_input(image_emb, means)
    assert fused_image_only.shape == (2, 1 + 2)
    # text portion (first column) should be the mean (15.0) for every row
    torch.testing.assert_close(fused_image_only[:, 0], torch.tensor([15.0, 15.0]))
    # image portion preserved exactly
    torch.testing.assert_close(fused_image_only[:, 1:], image_emb)


def test_build_text_only_input_replaces_image_with_mean():
    image_emb = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    text_emb = torch.tensor([[10.0], [20.0]])
    means = compute_modality_means(image_emb, text_emb)

    fused_text_only = build_text_only_input(text_emb, means)
    assert fused_text_only.shape == (2, 1 + 2)
    torch.testing.assert_close(fused_text_only[:, :1], text_emb)
    # image portion should be the mean [2.0, 3.0] for every row
    torch.testing.assert_close(fused_text_only[:, 1], torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(fused_text_only[:, 2], torch.tensor([3.0, 3.0]))


def test_no_zero_vectors_used():
    image_emb = torch.ones(5, 3) * 7.0
    text_emb = torch.ones(5, 2) * 9.0
    means = compute_modality_means(image_emb, text_emb)
    fused = build_image_only_input(image_emb, means)
    assert not torch.any(fused == 0.0)  # never falls back to a zero placeholder
