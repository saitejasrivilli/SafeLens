"""Trains only the classification head on cached frozen-CLIP embeddings.
Model selection uses validation (dev) PR-AUC by default -- never test.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
from torch import nn

from safelens.models.text.deberta.train import compute_class_weights
from safelens.models.text.metrics import compute_metrics
from safelens.models.vision.clip.config import HeadConfig, ImageTrainingConfig
from safelens.models.vision.clip.head import CLIP_EMBED_DIM, ClassificationHead

TRACKING_THRESHOLD = 0.5  # fixed threshold used only to track metrics during training


@dataclass(frozen=True)
class TrainingResult:
    best_state_dict: dict[str, torch.Tensor]
    best_epoch: int
    epochs_run: int
    history: list[dict[str, float]] = field(default_factory=list)
    class_weights: list[float] | None = None


def train_head(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    dev_embeddings: torch.Tensor,
    dev_labels: torch.Tensor,
    config: ImageTrainingConfig,
    head_config: HeadConfig,
    embed_dim: int = CLIP_EMBED_DIM,
) -> TrainingResult:
    torch.manual_seed(config.seed)

    head = ClassificationHead(head_config, embed_dim=embed_dim)
    optimizer = torch.optim.Adam(
        head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    class_weights = (
        compute_class_weights(train_labels.tolist()) if config.use_class_weighting else None
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    n = train_embeddings.shape[0]
    best_metric = float("-inf")
    best_state: dict[str, torch.Tensor] = copy.deepcopy(head.state_dict())
    best_epoch = 0
    patience_counter = 0
    history: list[dict[str, float]] = []

    generator = torch.Generator().manual_seed(config.seed)

    epochs_run = 0
    for epoch in range(config.epochs):
        epochs_run = epoch + 1
        head.train()
        perm = torch.randperm(n, generator=generator)
        for i in range(0, n, config.batch_size):
            idx = perm[i : i + config.batch_size]
            batch_x = train_embeddings[idx]
            batch_y = train_labels[idx]

            optimizer.zero_grad()
            logits = head(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            dev_logits = head(dev_embeddings)
            dev_probs = torch.softmax(dev_logits, dim=-1)[:, 1].numpy()
        dev_metrics = compute_metrics(dev_labels.tolist(), dev_probs, TRACKING_THRESHOLD)
        selection_value = dev_metrics[config.model_selection_metric]
        history.append(
            {"epoch": epoch, "dev_pr_auc": dev_metrics["pr_auc"], "dev_f1": dev_metrics["f1"]}
        )

        if selection_value > best_metric:
            best_metric = selection_value
            best_state = copy.deepcopy(head.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                break

    return TrainingResult(
        best_state_dict=best_state,
        best_epoch=best_epoch,
        epochs_run=epochs_run,
        history=history,
        class_weights=class_weights.tolist() if class_weights is not None else None,
    )
