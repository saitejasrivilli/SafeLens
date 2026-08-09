"""Save/load Phase 5A (image-only) artifacts.

Layout:
  models/vision/clip/<model_version>/
    head.pt          # classification head state_dict only -- small, CLIP weights never saved
    config.json        # full ImageBaselineConfig used for this run
    threshold.json      # frozen decision threshold
    metadata.json        # dataset manifest reference, git commit, environment
"""

from __future__ import annotations

import json
import sys
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

import torch

from safelens.models.text.artifacts import git_commit
from safelens.models.vision.clip.config import ImageBaselineConfig
from safelens.utils.device import detect_device

MODELS_ROOT = Path(__file__).resolve().parents[5] / "models" / "vision" / "clip"


def artifact_dir(model_version: str, root: Path = MODELS_ROOT) -> Path:
    return root / model_version


def environment_metadata() -> dict[str, Any]:
    device_info = detect_device()
    packages = ["torch", "transformers", "scikit-learn"]
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
    head_state_dict: dict[str, torch.Tensor],
    config: ImageBaselineConfig,
    threshold: float,
    dataset_manifest_path: Path,
    training_metadata: dict[str, Any],
    root: Path = MODELS_ROOT,
) -> Path:
    out_dir = artifact_dir(model_version, root)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(head_state_dict, out_dir / "head.pt")
    (out_dir / "config.json").write_text(config.model_dump_json(indent=2))
    (out_dir / "threshold.json").write_text(json.dumps({"decision_threshold": threshold}))

    manifest = (
        json.loads(dataset_manifest_path.read_text()) if dataset_manifest_path.exists() else {}
    )
    metadata = {
        "dataset_manifest": manifest,
        "training_metadata": training_metadata,
        **environment_metadata(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    return out_dir


def load_artifacts(
    model_version: str, root: Path = MODELS_ROOT
) -> tuple[dict[str, torch.Tensor], ImageBaselineConfig]:
    out_dir = artifact_dir(model_version, root)
    head_state_dict = torch.load(out_dir / "head.pt", weights_only=True)
    config = ImageBaselineConfig.model_validate_json((out_dir / "config.json").read_text())
    return head_state_dict, config
