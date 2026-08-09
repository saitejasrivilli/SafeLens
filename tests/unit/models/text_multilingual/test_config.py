from pathlib import Path

from safelens.models.text_multilingual.config import load_text_arabic_config

SAMPLE_YAML = """
model_name: test-encoder
model_version: v1
hf_model_name: some/model
hf_model_revision: abc123
max_seq_length: 32
head:
  hidden_dim: 128
  dropout: 0.1
training:
  learning_rate: 0.001
  batch_size: 16
  epochs: 10
  use_class_weighting: true
decision_threshold:
  candidates: [0.1, 0.5, 0.9]
  selection_metric: f1
"""


def test_load_text_arabic_config(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(SAMPLE_YAML)

    config = load_text_arabic_config(path)
    assert config.model_name == "test-encoder"
    assert config.hf_model_name == "some/model"
    assert config.max_seq_length == 32
    assert config.head.hidden_dim == 128
    assert config.training.batch_size == 16
    assert config.decision_threshold.candidates == [0.1, 0.5, 0.9]


def test_config_defaults_when_omitted(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text("model_name: minimal\nhf_model_name: x\nhf_model_revision: y\n")

    config = load_text_arabic_config(path)
    assert config.model_version == "v1"
    assert config.max_seq_length == 64
    assert config.head.hidden_dim == 256
