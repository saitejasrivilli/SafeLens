"""Deterministic tokenization + frozen text-encoder embedding extraction.

Deliberately has NO access to `MultimodalExample.image_path` anywhere in
this module -- the text-only experiment must never use image or filename
information, during either training or inference.
"""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from safelens.data.multimodal.schema import MultimodalExample
from safelens.models.text_multilingual.encoder import encode_texts


@torch.no_grad()
def extract_embeddings(
    examples: list[MultimodalExample],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    device: str,
    max_seq_length: int,
    batch_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Batched, deterministic (fixed truncation/padding, eval-mode encoder,
    no augmentation) embedding extraction. Returns
    (embeddings [N, hidden_size], labels [N], example_ids)."""
    all_embeddings: list[torch.Tensor] = []
    all_labels: list[int] = []
    all_ids: list[str] = []

    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        texts = [ex.text for ex in batch]
        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
            return_tensors="pt",
        )
        embeddings = encode_texts(
            model, encodings["input_ids"], encodings["attention_mask"], device
        )

        all_embeddings.append(embeddings)
        all_labels.extend(ex.hate_label for ex in batch)
        all_ids.extend(ex.example_id for ex in batch)

    return torch.cat(all_embeddings, dim=0), torch.tensor(all_labels, dtype=torch.long), all_ids
