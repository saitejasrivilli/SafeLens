"""Configuration for the frozen CLIP+AraBERT fusion model (Phase 5C).
Reuses HeadConfig/TrainingConfig/DecisionThresholdConfig -- the fusion
head, training loop, and threshold logic are architecture-agnostic
(operate on a single concatenated embedding tensor), so this is the same
config shape as Phase 5A/5B, just with two encoder identities instead of
one."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from safelens.models.text.config import DecisionThresholdConfig
from safelens.models.vision.clip.config import HeadConfig
from safelens.models.vision.clip.config import ImageTrainingConfig as TrainingConfig

CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs" / "multimodal_fusion.yaml"


class FusionConfig(BaseModel):
    model_name: str = "fusion-clip-arabert"
    model_version: str = "v1"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_revision: str = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
    text_model_name: str = "aubmindlab/bert-base-arabertv2"
    text_model_revision: str = "97522efce17efa33036ac619802d5cec238dcad9"
    max_seq_length: int = 64
    head: HeadConfig = HeadConfig()
    training: TrainingConfig = TrainingConfig()
    decision_threshold: DecisionThresholdConfig = DecisionThresholdConfig()


def load_fusion_config(path: Path | None = None) -> FusionConfig:
    path = path or CONFIG_PATH
    raw = yaml.safe_load(path.read_text())
    return FusionConfig(**raw)
