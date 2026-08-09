from safelens.models.multimodal.config import load_fusion_config


def test_load_fusion_config_defaults():
    config = load_fusion_config()
    assert config.clip_model_name == "openai/clip-vit-base-patch32"
    assert config.text_model_name == "aubmindlab/bert-base-arabertv2"
    assert config.head.hidden_dim == 256
    assert config.training.use_class_weighting is True
    assert 0.5 in config.decision_threshold.candidates
