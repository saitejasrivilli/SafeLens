"""Configuration for the TF-IDF + Logistic Regression baseline."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs" / "baseline.yaml"


class TfidfConfig(BaseModel):
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    max_features: int = 20000
    sublinear_tf: bool = True


class LogisticRegressionConfig(BaseModel):
    C: float = 1.0
    class_weight: str | None = "balanced"
    max_iter: int = 1000
    random_state: int = 42


class LabelConfig(BaseModel):
    target: str = "toxicity"
    ground_truth_threshold: float = 0.5


class DecisionThresholdConfig(BaseModel):
    candidates: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    selection_metric: str = "f1"


class BaselineConfig(BaseModel):
    model_name: str = "baseline-tfidf-logreg"
    model_version: str = "v1"
    tfidf: TfidfConfig = TfidfConfig()
    logistic_regression: LogisticRegressionConfig = LogisticRegressionConfig()
    label: LabelConfig = LabelConfig()
    decision_threshold: DecisionThresholdConfig = DecisionThresholdConfig()


def load_baseline_config(path: Path | None = None) -> BaselineConfig:
    path = path or CONFIG_PATH
    raw = yaml.safe_load(path.read_text())
    raw["tfidf"]["ngram_range"] = tuple(raw["tfidf"]["ngram_range"])
    return BaselineConfig(**raw)
