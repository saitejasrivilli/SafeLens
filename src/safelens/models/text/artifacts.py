"""Save/load baseline model artifacts with explicit versioning.

Layout:
  models/baseline/<model_version>/
    vectorizer.joblib
    model.joblib
    config.json          # full BaselineConfig used for this run
    metadata.json        # dataset version/hash, git commit, python/pkg versions
"""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

import joblib

from safelens.models.text.config import BaselineConfig

MODELS_ROOT = Path(__file__).resolve().parents[4] / "models" / "baseline"


def artifact_dir(model_version: str, root: Path = MODELS_ROOT) -> Path:
    return root / model_version


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def environment_metadata() -> dict[str, Any]:
    packages = ["scikit-learn", "scipy", "numpy", "pydantic"]
    return {
        "python_version": sys.version,
        "packages": {p: pkg_version(p) for p in packages},
        "git_commit": git_commit(),
    }


def save_artifacts(
    *,
    model_version: str,
    vectorizer,
    model,
    config: BaselineConfig,
    dataset_version: str,
    dataset_manifest_path: Path,
    root: Path = MODELS_ROOT,
) -> Path:
    out_dir = artifact_dir(model_version, root)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, out_dir / "vectorizer.joblib")
    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "config.json").write_text(config.model_dump_json(indent=2))

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


def load_artifacts(model_version: str, root: Path = MODELS_ROOT) -> tuple[Any, Any, BaselineConfig]:
    out_dir = artifact_dir(model_version, root)
    vectorizer = joblib.load(out_dir / "vectorizer.joblib")
    model = joblib.load(out_dir / "model.joblib")
    config = BaselineConfig.model_validate_json((out_dir / "config.json").read_text())
    return vectorizer, model, config
