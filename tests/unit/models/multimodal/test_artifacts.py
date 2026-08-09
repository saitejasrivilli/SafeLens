import json
from pathlib import Path

import torch

from safelens.models.multimodal.artifacts import artifact_dir, save_artifacts
from safelens.models.multimodal.config import FusionConfig
from safelens.models.vision.clip.head import ClassificationHead


def test_save_artifacts_writes_expected_files(tmp_path: Path):
    config = FusionConfig(model_version="test-v1")
    head = ClassificationHead(config.head, embed_dim=1280)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"dataset_name": "fake"}))

    out_dir = save_artifacts(
        model_version="test-v1",
        head_state_dict=head.state_dict(),
        config=config,
        threshold=0.5,
        dataset_manifest_path=manifest_path,
        training_metadata={"epochs_run": 4},
        root=tmp_path / "models",
    )

    assert out_dir == artifact_dir("test-v1", tmp_path / "models")
    assert (out_dir / "head.pt").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "threshold.json").exists()
    assert (out_dir / "metadata.json").exists()

    metadata = json.loads((out_dir / "metadata.json").read_text())
    assert metadata["training_metadata"]["epochs_run"] == 4

    loaded_state = torch.load(out_dir / "head.pt", weights_only=True)
    reloaded = ClassificationHead(config.head, embed_dim=1280)
    reloaded.load_state_dict(loaded_state)
