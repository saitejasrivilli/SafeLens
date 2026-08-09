"""Configuration for the DeBERTa-v3-small fine-tune. Reuses LabelConfig and
DecisionThresholdConfig from the baseline config so the two experiments
share the identical binarization/threshold-selection semantics."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from safelens.models.text.config import DecisionThresholdConfig, LabelConfig

CONFIG_PATH = Path(__file__).resolve().parents[5] / "configs" / "deberta.yaml"


class TokenizerConfig(BaseModel):
    max_seq_length: int = 256


class TrainingConfig(BaseModel):
    learning_rate: float = 2.0e-5
    train_batch_size: int = 16
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 2
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 42
    use_class_weighting: bool = True
    early_stopping_patience: int = 2
    model_selection_metric: str = "pr_auc"


class DebertaConfig(BaseModel):
    model_name: str = "deberta-v3-small-toxicity"
    model_version: str = "v1"
    hf_model_name: str = "microsoft/deberta-v3-small"
    hf_model_revision: str = "a36c739020e01763fe789b4b85e2df55d6180012"
    tokenizer: TokenizerConfig = TokenizerConfig()
    training: TrainingConfig = TrainingConfig()
    label: LabelConfig = LabelConfig()
    decision_threshold: DecisionThresholdConfig = DecisionThresholdConfig()


def load_deberta_config(path: Path | None = None) -> DebertaConfig:
    path = path or CONFIG_PATH
    raw = yaml.safe_load(path.read_text())
    return DebertaConfig(**raw)
