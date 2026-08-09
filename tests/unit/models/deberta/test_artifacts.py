import json
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from safelens.models.text.deberta.artifacts import artifact_dir, save_artifacts
from safelens.models.text.deberta.config import DebertaConfig

TINY_MODEL = "ydshieh/tiny-random-DebertaV2ForSequenceClassification"


def test_save_artifacts_writes_expected_files(tmp_path: Path, tiny_model, tiny_tokenizer):
    config = DebertaConfig(model_version="test-v1")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"dataset_name": "fake"}))

    out_dir = save_artifacts(
        model_version="test-v1",
        model=tiny_model,
        tokenizer=tiny_tokenizer,
        config=config,
        threshold=0.42,
        dataset_version="fake@0-4",
        dataset_manifest_path=manifest_path,
        root=tmp_path / "models",
    )

    assert out_dir == artifact_dir("test-v1", tmp_path / "models")
    assert (out_dir / "model" / "config.json").exists()
    assert (out_dir / "tokenizer").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "threshold.json").exists()

    threshold_data = json.loads((out_dir / "threshold.json").read_text())
    assert threshold_data["decision_threshold"] == 0.42

    metadata = json.loads((out_dir / "metadata.json").read_text())
    assert metadata["dataset_version"] == "fake@0-4"
    assert metadata["dataset_manifest"]["dataset_name"] == "fake"
    assert "git_commit" in metadata
    assert "device" in metadata


def test_saved_model_reloads_with_same_weights(tmp_path: Path, tiny_model, tiny_tokenizer):
    config = DebertaConfig(model_version="test-v2")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")

    out_dir = save_artifacts(
        model_version="test-v2",
        model=tiny_model,
        tokenizer=tiny_tokenizer,
        config=config,
        threshold=0.5,
        dataset_version="fake@0-4",
        dataset_manifest_path=manifest_path,
        root=tmp_path / "models",
    )
    reloaded = AutoModelForSequenceClassification.from_pretrained(out_dir / "model")
    reloaded_tokenizer = AutoTokenizer.from_pretrained(out_dir / "tokenizer")
    assert reloaded.config.num_labels == 2
    assert reloaded_tokenizer.vocab_size == tiny_tokenizer.vocab_size
