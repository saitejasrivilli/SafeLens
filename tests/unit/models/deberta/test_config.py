from safelens.models.text.deberta.config import load_deberta_config


def test_load_deberta_config_defaults():
    config = load_deberta_config()
    assert config.hf_model_name == "microsoft/deberta-v3-small"
    assert config.hf_model_revision
    assert config.tokenizer.max_seq_length == 256
    assert config.training.use_class_weighting is True
    assert config.label.target == "toxicity"
    assert config.label.ground_truth_threshold == 0.5
    assert 0.5 in config.decision_threshold.candidates
