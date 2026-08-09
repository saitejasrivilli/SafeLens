"""DeBERTa-v3-small fine-tuning: model/tokenizer construction, class-weighted
loss, and the Trainer-compatible metrics callback used for model selection.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)

from safelens.models.text.deberta.config import DebertaConfig
from safelens.models.text.metrics import compute_metrics

MODEL_SELECTION_THRESHOLD = (
    0.5  # fixed threshold used only for tracking eval_f1 etc during training
)


def build_model_and_tokenizer(
    config: DebertaConfig,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    tokenizer = AutoTokenizer.from_pretrained(
        config.hf_model_name, revision=config.hf_model_revision
    )
    # The published checkpoint stores weights in float16. Training in float16
    # without a mixed-precision scaler (not used here -- fp16/bf16 are only
    # enabled for CUDA, see docs/model_design.md) is numerically unstable and
    # also fails outright on MPS/CPU (dtype mismatch against float32 loss
    # inputs), so the model is forced to float32 for training.
    model = AutoModelForSequenceClassification.from_pretrained(
        config.hf_model_name,
        revision=config.hf_model_revision,
        num_labels=2,
        torch_dtype=torch.float32,
    )
    return tokenizer, model


def compute_class_weights(labels: list[int]) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss. [w_negative, w_positive]."""
    labels_t = torch.tensor(labels, dtype=torch.long)
    counts = torch.bincount(labels_t, minlength=2).float()
    total = counts.sum()
    weights = total / (2.0 * counts.clamp(min=1))
    return weights


def build_compute_metrics_fn():
    def _compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        logits = eval_pred.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.softmax(torch.from_numpy(np.asarray(logits)), dim=-1)[:, 1].numpy()
        labels = eval_pred.label_ids
        if isinstance(labels, tuple):
            labels = labels[0]
        metrics = compute_metrics(labels, probs, MODEL_SELECTION_THRESHOLD)
        # Trainer only accepts flat scalar metrics.
        return {
            "f1": metrics["f1"],
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "false_positive_rate": metrics["false_positive_rate"],
            "false_negative_rate": metrics["false_negative_rate"],
        }

    return _compute_metrics


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss. Weighting is
    configurable (`training.use_class_weighting`) -- see docs/model_design.md
    for why it is not assumed automatically optimal."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
