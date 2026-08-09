import json
from pathlib import Path

import torch

from safelens.models.text_multilingual.artifacts import artifact_dir, save_artifacts
from safelens.models.text_multilingual.config import TextArabicConfig
from safelens.models.vision.clip.config import HeadConfig
from safelens.models.vision.clip.head import ClassificationHead


def test_save_artifacts_writes_expected_files(tmp_path: Path):
    config = TextArabicConfig(
        model_name="test-encoder",
        model_version="v1",
        hf_model_name="some/model",
        hf_model_revision="abc123",
        head=HeadConfig(hidden_dim=8),
    )
    head = ClassificationHead(config.head, embed_dim=32)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"dataset_name": "fake"}))

    out_dir = save_artifacts(
        head_state_dict=head.state_dict(),
        config=config,
        threshold=0.5,
        dataset_manifest_path=manifest_path,
        training_metadata={"epochs_run": 3},
        root=tmp_path / "models",
    )

    assert out_dir == artifact_dir("test-encoder", "v1", tmp_path / "models")
    assert (out_dir / "head.pt").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "threshold.json").exists()
    assert (out_dir / "metadata.json").exists()

    metadata = json.loads((out_dir / "metadata.json").read_text())
    assert metadata["dataset_manifest"]["dataset_name"] == "fake"
    assert metadata["training_metadata"]["epochs_run"] == 3
    assert "git_commit" in metadata

    loaded_state = torch.load(out_dir / "head.pt", weights_only=True)
    reloaded = ClassificationHead(config.head, embed_dim=32)
    reloaded.load_state_dict(loaded_state)
