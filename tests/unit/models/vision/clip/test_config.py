from safelens.models.vision.clip.config import load_image_baseline_config


def test_load_image_baseline_config_defaults():
    config = load_image_baseline_config()
    assert config.clip_model_name == "openai/clip-vit-base-patch32"
    assert config.clip_revision
    assert config.head.hidden_dim == 256
    assert config.training.use_class_weighting is True
    assert config.training.model_selection_metric == "pr_auc"
    assert 0.5 in config.decision_threshold.candidates
