"""Loads the Phase 2 processed, leakage-safe splits. Does not modify them."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[4] / "data" / "processed" / "civil_comments"


@dataclass(frozen=True)
class SplitData:
    content_ids: list[str]
    texts: list[str]
    scores: list[float]  # continuous label score for `label.target`
    dataset_version: str


def load_split(split: str, target: str, processed_dir: Path = PROCESSED_DIR) -> SplitData:
    """split: one of "train", "validation", "test" as written by
    scripts/prepare_data.py. Raises FileNotFoundError if Phase 2 has not
    been run — this module never (re)generates the split itself."""
    path = processed_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make data-download && make data-prepare` first "
            "(Phase 2) before training the baseline."
        )

    content_ids: list[str] = []
    texts: list[str] = []
    scores: list[float] = []
    dataset_version = ""

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        content_ids.append(row["content_id"])
        texts.append(row["text"])
        scores.append(row["labels"][target])
        dataset_version = row["dataset_version"]

    return SplitData(
        content_ids=content_ids, texts=texts, scores=scores, dataset_version=dataset_version
    )


def binarize(scores: list[float], threshold: float) -> list[int]:
    return [int(s >= threshold) for s in scores]
