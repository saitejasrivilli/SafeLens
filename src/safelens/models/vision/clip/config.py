"""Configuration for the frozen-CLIP image-only baseline (Phase 5A).

Reuses DecisionThresholdConfig from the text baseline config -- the
threshold-sweep/selection semantics are identical, no need to duplicate.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from safelens.models.text.config import DecisionThresholdConfig

CONFIG_PATH = Path(__file__).resolve().parents[5] / "configs" / "image_baseline.yaml"


class HeadConfig(BaseModel):
    hidden_dim: int = 256
    dropout: float = 0.2


class ImageTrainingConfig(BaseModel):
    learning_rate: float = 1.0e-3
    batch_size: int = 64
    epochs: int = 50
    weight_decay: float = 0.01
    seed: int = 42
    use_class_weighting: bool = True
    early_stopping_patience: int = 8
    model_selection_metric: str = "pr_auc"


class ImageBaselineConfig(BaseModel):
    model_name: str = "image-only-clip-vit-b32"
    model_version: str = "v1"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_revision: str = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
    head: HeadConfig = HeadConfig()
    training: ImageTrainingConfig = ImageTrainingConfig()
    decision_threshold: DecisionThresholdConfig = DecisionThresholdConfig()


def load_image_baseline_config(path: Path | None = None) -> ImageBaselineConfig:
    path = path or CONFIG_PATH
    raw = yaml.safe_load(path.read_text())
    return ImageBaselineConfig(**raw)
