"""Missing-modality representation for inference-time ablation.

Design decision (documented, not silently chosen): a missing modality's
embedding is replaced with the **mean embedding vector of that modality
over the training set**, not a zero vector. A zero vector is
out-of-distribution for a frozen encoder's output space (CLIP/AraBERT
embeddings are never actually all-zero for real input) and could push the
fusion head into an arbitrary, untrained region of its input space,
confounding the ablation. The train-mean vector is a simple, standard
"neutral/uninformative" placeholder -- it represents "the average signal,
carrying no example-specific information" rather than an artificial
out-of-distribution point. This is NOT a learned gating mechanism (that
would require joint training, out of scope for a frozen-encoder-plus-head
baseline) -- it is an explicit, documented imputation choice for this
ablation only, never used during normal training or inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModalityMeans:
    image_mean: torch.Tensor  # [image_dim]
    text_mean: torch.Tensor  # [text_dim]


def compute_modality_means(
    train_image_embeddings: torch.Tensor, train_text_embeddings: torch.Tensor
) -> ModalityMeans:
    return ModalityMeans(
        image_mean=train_image_embeddings.mean(dim=0),
        text_mean=train_text_embeddings.mean(dim=0),
    )


def build_image_only_input(image_embeddings: torch.Tensor, means: ModalityMeans) -> torch.Tensor:
    """Fusion-model input with the text side replaced by the train-mean
    text vector (image information only reaches the fusion head)."""
    n = image_embeddings.shape[0]
    text_placeholder = means.text_mean.unsqueeze(0).expand(n, -1)
    return torch.cat([text_placeholder, image_embeddings], dim=1)


def build_text_only_input(text_embeddings: torch.Tensor, means: ModalityMeans) -> torch.Tensor:
    """Fusion-model input with the image side replaced by the train-mean
    image vector (text information only reaches the fusion head)."""
    n = text_embeddings.shape[0]
    image_placeholder = means.image_mean.unsqueeze(0).expand(n, -1)
    return torch.cat([text_embeddings, image_placeholder], dim=1)
