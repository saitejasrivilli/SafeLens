from safelens.models.text.config import load_baseline_config


def test_load_baseline_config_defaults():
    config = load_baseline_config()
    assert config.tfidf.ngram_range == (1, 2)
    assert config.logistic_regression.class_weight == "balanced"
    assert config.label.target == "toxicity"
    assert 0.1 in config.decision_threshold.candidates
    assert config.label.ground_truth_threshold == 0.5
