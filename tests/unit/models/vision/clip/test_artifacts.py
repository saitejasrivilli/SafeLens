import json
from pathlib import Path

import torch

from safelens.models.vision.clip.artifacts import load_artifacts, save_artifacts
from safelens.models.vision.clip.config import HeadConfig, ImageBaselineConfig
from safelens.models.vision.clip.head import ClassificationHead


def test_save_and_load_artifacts_round_trip(tmp_path: Path):
    config = ImageBaselineConfig(model_version="test-v1", head=HeadConfig(hidden_dim=8))
    head = ClassificationHead(config.head, embed_dim=64)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"dataset_name": "fake"}))

    out_dir = save_artifacts(
        model_version="test-v1",
        head_state_dict=head.state_dict(),
        config=config,
        threshold=0.42,
        dataset_manifest_path=manifest_path,
        training_metadata={"epochs_run": 5},
        root=tmp_path / "models",
    )

    assert (out_dir / "head.pt").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "threshold.json").exists()
    assert (out_dir / "metadata.json").exists()

    threshold_data = json.loads((out_dir / "threshold.json").read_text())
    assert threshold_data["decision_threshold"] == 0.42

    metadata = json.loads((out_dir / "metadata.json").read_text())
    assert metadata["dataset_manifest"]["dataset_name"] == "fake"
    assert metadata["training_metadata"]["epochs_run"] == 5
    assert "git_commit" in metadata

    loaded_state_dict, loaded_config = load_artifacts("test-v1", root=tmp_path / "models")
    assert loaded_config.model_version == "test-v1"

    reloaded_head = ClassificationHead(loaded_config.head, embed_dim=64)
    reloaded_head.load_state_dict(loaded_state_dict)

    x = torch.randn(2, 64)
    head.eval()
    reloaded_head.eval()
    torch.testing.assert_close(head(x), reloaded_head(x))
