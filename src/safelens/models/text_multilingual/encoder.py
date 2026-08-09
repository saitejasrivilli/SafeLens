"""Frozen Arabic/multilingual text encoder. Entire encoder is frozen --
only the classification head is ever trained. Uses mean-pooling over the
last hidden state (masked by attention_mask), the standard approach for
turning a base encoder's token embeddings into one sentence embedding when
no dedicated pooler/projection head is used.
"""

from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from safelens.models.text_multilingual.config import TextArabicConfig


def build_text_encoder(
    config: TextArabicConfig,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    tokenizer = AutoTokenizer.from_pretrained(
        config.hf_model_name, revision=config.hf_model_revision
    )
    model = AutoModel.from_pretrained(config.hf_model_name, revision=config.hf_model_revision)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return tokenizer, model


@torch.no_grad()
def encode_texts(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Deterministic (eval mode, no dropout), masked mean-pooled sentence
    embedding, shape (B, hidden_size)."""
    model = model.to(device)  # type: ignore[arg-type]  # PreTrainedModel.to() str overload
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    token_embeddings = outputs.last_hidden_state  # (B, L, H)
    mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    pooled = summed / counts
    return pooled.cpu()
