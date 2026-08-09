"""Frozen CLIP image encoder. The entire CLIP model is frozen -- only the
classification head (head.py) is ever trained. Uses the explicit
vision_model + visual_projection path rather than `get_image_features`,
since the latter returns an unprojected pooled output in the installed
transformers version (verified directly, not assumed) -- this path is
deterministic and matches CLIP's published 512-dim joint embedding space.
"""

from __future__ import annotations

import torch
from transformers import CLIPImageProcessor, CLIPModel

from safelens.models.vision.clip.config import ImageBaselineConfig


def build_clip_encoder(
    config: ImageBaselineConfig,
) -> tuple[CLIPImageProcessor, CLIPModel]:
    processor = CLIPImageProcessor.from_pretrained(
        config.clip_model_name, revision=config.clip_revision
    )
    model = CLIPModel.from_pretrained(config.clip_model_name, revision=config.clip_revision)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return processor, model


@torch.no_grad()
def encode_images(model: CLIPModel, pixel_values: torch.Tensor, device: str) -> torch.Tensor:
    """pixel_values: already-preprocessed batch, shape (B, 3, H, W).
    Returns the projected 512-dim image embedding, deterministic (eval
    mode, no dropout/augmentation in the frozen encoder)."""
    model = model.to(device)  # type: ignore[arg-type]  # PreTrainedModel.to() str overload
    pixel_values = pixel_values.to(device)
    vision_out = model.vision_model(pixel_values=pixel_values)
    projected = model.visual_projection(vision_out.pooler_output)
    return projected.cpu()
