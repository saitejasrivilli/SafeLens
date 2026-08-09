"""Configuration for the frozen Arabic/multilingual text-encoder baselines
(Phase 5B). Reuses HeadConfig and the training-loop config from the Phase 5A
vision module -- the classification head and training loop are
architecture-agnostic (operate on cached embedding tensors regardless of
whether the embedding came from CLIP or a text encoder), so duplicating
those config classes here would just be a model zoo, not a real
difference. Also reuses DecisionThresholdConfig from the text baseline
config -- same threshold-sweep semantics everywhere in this repo.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from safelens.models.text.config import DecisionThresholdConfig
from safelens.models.vision.clip.config import HeadConfig
from safelens.models.vision.clip.config import ImageTrainingConfig as TrainingConfig


class TextArabicConfig(BaseModel):
    model_name: str
    model_version: str = "v1"
    hf_model_name: str
    hf_model_revision: str
    max_seq_length: int = 64
    head: HeadConfig = HeadConfig()
    training: TrainingConfig = TrainingConfig()
    decision_threshold: DecisionThresholdConfig = DecisionThresholdConfig()


def load_text_arabic_config(path: Path) -> TextArabicConfig:
    raw = yaml.safe_load(path.read_text())
    return TextArabicConfig(**raw)
