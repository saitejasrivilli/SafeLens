"""Deterministic image loading + frozen-CLIP embedding extraction.

Deliberately has NO access to `MultimodalExample.text` anywhere in this
module -- the image-only experiment must never see the text field, during
either training or inference.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel

from safelens.data.multimodal.schema import MultimodalExample
from safelens.models.vision.clip.encoder import encode_images


def _load_image(raw_image_root: Path, example: MultimodalExample) -> Image.Image:
    return Image.open(raw_image_root / example.image_path).convert("RGB")


@torch.no_grad()
def extract_embeddings(
    examples: list[MultimodalExample],
    raw_image_root: Path,
    processor: CLIPImageProcessor,
    model: CLIPModel,
    device: str,
    batch_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Batched, deterministic (no augmentation, eval-mode encoder) embedding
    extraction. Returns (embeddings [N, 512], labels [N], example_ids)."""
    all_embeddings: list[torch.Tensor] = []
    all_labels: list[int] = []
    all_ids: list[str] = []

    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        images = [_load_image(raw_image_root, ex) for ex in batch]
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
        embeddings = encode_images(model, pixel_values, device)

        all_embeddings.append(embeddings)
        all_labels.extend(ex.hate_label for ex in batch)
        all_ids.extend(ex.example_id for ex in batch)

    return torch.cat(all_embeddings, dim=0), torch.tensor(all_labels, dtype=torch.long), all_ids
