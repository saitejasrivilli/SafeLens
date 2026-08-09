"""Extracts frozen CLIP image embeddings and frozen AraBERT text embeddings
for the same examples, in the same order, and concatenates them into one
fused embedding per example. Only reads `.image_path` (resolved against the
raw image root) and `.text` -- never filename/account/dataset-ID features
beyond what's needed to locate the image file on disk, per the explicit
"no metadata as a predictive feature" instruction: the file path is I/O
plumbing, not a model input; nothing derived from it is concatenated into
the embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    CLIPModel,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from safelens.data.multimodal.schema import MultimodalExample
from safelens.models.text_multilingual.encoder import encode_texts
from safelens.models.vision.clip.encoder import encode_images


@dataclass(frozen=True)
class FusedEmbeddings:
    image_embeddings: torch.Tensor  # [N, image_dim]
    text_embeddings: torch.Tensor  # [N, text_dim]
    fused_embeddings: torch.Tensor  # [N, image_dim + text_dim]
    labels: torch.Tensor  # [N]
    example_ids: list[str]


@torch.no_grad()
def extract_fused_embeddings(
    examples: list[MultimodalExample],
    raw_image_root: Path,
    image_processor: CLIPImageProcessor,
    image_model: CLIPModel,
    text_tokenizer: PreTrainedTokenizerBase,
    text_model: PreTrainedModel,
    device: str,
    max_seq_length: int,
    batch_size: int = 32,
) -> FusedEmbeddings:
    image_embeddings: list[torch.Tensor] = []
    text_embeddings: list[torch.Tensor] = []
    labels: list[int] = []
    ids: list[str] = []

    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]

        images = [Image.open(raw_image_root / ex.image_path).convert("RGB") for ex in batch]
        pixel_values = image_processor(images=images, return_tensors="pt")["pixel_values"]
        img_emb = encode_images(image_model, pixel_values, device)

        texts = [ex.text for ex in batch]
        encodings = text_tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
            return_tensors="pt",
        )
        txt_emb = encode_texts(
            text_model, encodings["input_ids"], encodings["attention_mask"], device
        )

        image_embeddings.append(img_emb)
        text_embeddings.append(txt_emb)
        labels.extend(ex.hate_label for ex in batch)
        ids.extend(ex.example_id for ex in batch)

    image_tensor = torch.cat(image_embeddings, dim=0)
    text_tensor = torch.cat(text_embeddings, dim=0)
    fused = torch.cat([text_tensor, image_tensor], dim=1)

    return FusedEmbeddings(
        image_embeddings=image_tensor,
        text_embeddings=text_tensor,
        fused_embeddings=fused,
        labels=torch.tensor(labels, dtype=torch.long),
        example_ids=ids,
    )
