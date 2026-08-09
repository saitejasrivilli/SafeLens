import math

import torch

from safelens.models.vision.clip.encoder import encode_images


def test_encode_images_shape_and_determinism(tiny_clip_model, tiny_clip_processor, sample_image):
    inputs = tiny_clip_processor(images=sample_image, return_tensors="pt")
    proj_dim = tiny_clip_model.config.projection_dim

    emb_a = encode_images(tiny_clip_model, inputs["pixel_values"], "cpu")
    emb_b = encode_images(tiny_clip_model, inputs["pixel_values"], "cpu")

    assert emb_a.shape == (1, proj_dim)
    assert torch.equal(emb_a, emb_b)  # frozen, eval mode -> deterministic
    assert not torch.isnan(emb_a).any()
    assert all(math.isfinite(v) for v in emb_a.flatten().tolist())


def test_encoder_params_are_frozen(tiny_clip_model):
    assert all(not p.requires_grad for p in tiny_clip_model.parameters())


def test_encode_images_batches_independently(tiny_clip_model, tiny_clip_processor):
    from PIL import Image

    img_a = Image.new("RGB", (64, 64), (255, 0, 0))
    img_b = Image.new("RGB", (64, 64), (0, 255, 0))

    inputs = tiny_clip_processor(images=[img_a, img_b], return_tensors="pt")
    batch_emb = encode_images(tiny_clip_model, inputs["pixel_values"], "cpu")

    single_a = tiny_clip_processor(images=img_a, return_tensors="pt")
    emb_a = encode_images(tiny_clip_model, single_a["pixel_values"], "cpu")

    assert batch_emb.shape[0] == 2
    torch.testing.assert_close(batch_emb[0], emb_a[0])
