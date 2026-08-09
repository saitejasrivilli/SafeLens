"""Lightweight classification head trained on frozen CLIP embeddings.
Deliberately minimal: Linear -> ReLU -> Dropout -> Linear -> 2-class logits.
"""

from __future__ import annotations

import torch
from torch import nn

from safelens.models.vision.clip.config import HeadConfig

CLIP_EMBED_DIM = 512  # openai/clip-vit-base-patch32 projection_dim


class ClassificationHead(nn.Module):
    def __init__(self, config: HeadConfig, embed_dim: int = CLIP_EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 2),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)
