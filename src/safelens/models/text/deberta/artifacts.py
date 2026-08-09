"""Save/load DeBERTa experiment artifacts.

Layout:
  models/text/deberta/<model_version>/
    model/            # HF save_pretrained() output -- gitignored, not committed
    tokenizer/         # HF save_pretrained() output -- gitignored, not committed
    config.json        # full DebertaConfig used for this run -- small, could be committed
    metadata.json       # dataset version/hash, git commit, python/pkg/device info
    threshold.json       # frozen decision threshold
"""

from __future__ import annotations

import json
import sys
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

import torch

from safelens.models.text.artifacts import git_commit
from safelens.models.text.deberta.config import DebertaConfig
from safelens.utils.device import detect_device

MODELS_ROOT = Path(__file__).resolve().parents[5] / "models" / "text" / "deberta"


def artifact_dir(model_version: str, root: Path = MODELS_ROOT) -> Path:
    return root / model_version


def environment_metadata() -> dict[str, Any]:
    device_info = detect_device()
    packages = ["torch", "transformers", "accelerate", "scikit-learn"]
    return {
        "python_version": sys.version,
        "packages": {p: pkg_version(p) for p in packages},
        "git_commit": git_commit(),
        "device": device_info.device,
        "device_reason": device_info.reason,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }


def save_artifacts(
    *,
    model_version: str,
    model,
    tokenizer,
    config: DebertaConfig,
    threshold: float,
    dataset_version: str,
    dataset_manifest_path: Path,
    root: Path = MODELS_ROOT,
) -> Path:
    out_dir = artifact_dir(model_version, root)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(out_dir / "model")
    tokenizer.save_pretrained(out_dir / "tokenizer")

    (out_dir / "config.json").write_text(config.model_dump_json(indent=2))
    (out_dir / "threshold.json").write_text(json.dumps({"decision_threshold": threshold}))

    manifest = (
        json.loads(dataset_manifest_path.read_text()) if dataset_manifest_path.exists() else {}
    )
    metadata = {
        "dataset_version": dataset_version,
        "dataset_manifest": manifest,
        **environment_metadata(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    return out_dir
