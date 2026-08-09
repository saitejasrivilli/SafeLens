import math

import numpy as np
import pytest

from safelens.models.text.config import LogisticRegressionConfig, TfidfConfig
from safelens.models.text.pipeline import fit_model, fit_vectorizer

TRAIN_TEXTS = [
    "you are a wonderful person",
    "have a great day my friend",
    "i hate you and everyone like you",
    "this is a threat you will regret this",
    "thanks for the help today",
    "you are stupid and worthless",
]
TRAIN_LABELS = [0, 0, 1, 1, 0, 1]


def test_vectorizer_fits_only_on_training_vocabulary():
    config = TfidfConfig(min_df=1)
    vectorizer = fit_vectorizer(TRAIN_TEXTS, config)
    # word that only appears in a held-out sentence, never in TRAIN_TEXTS
    assert "zephyr" not in vectorizer.vocabulary_
    # word that does appear in training text is present
    assert "wonderful" in vectorizer.vocabulary_


def test_vectorizer_transform_does_not_mutate_vocabulary():
    config = TfidfConfig(min_df=1)
    vectorizer = fit_vectorizer(TRAIN_TEXTS, config)
    vocab_before = dict(vectorizer.vocabulary_)
    vectorizer.transform(["a completely novel sentence with zephyr unicorn"])
    assert vectorizer.vocabulary_ == vocab_before


def test_training_is_deterministic():
    config = TfidfConfig(min_df=1)
    lr_config = LogisticRegressionConfig(random_state=42, max_iter=1000)

    vec_a = fit_vectorizer(TRAIN_TEXTS, config)
    model_a = fit_model(vec_a.transform(TRAIN_TEXTS), TRAIN_LABELS, lr_config)

    vec_b = fit_vectorizer(TRAIN_TEXTS, config)
    model_b = fit_model(vec_b.transform(TRAIN_TEXTS), TRAIN_LABELS, lr_config)

    np.testing.assert_array_almost_equal(model_a.coef_, model_b.coef_)


def test_predict_proba_in_valid_range_and_finite():
    config = TfidfConfig(min_df=1)
    lr_config = LogisticRegressionConfig(random_state=42)
    vectorizer = fit_vectorizer(TRAIN_TEXTS, config)
    model = fit_model(vectorizer.transform(TRAIN_TEXTS), TRAIN_LABELS, lr_config)

    probs = model.predict_proba(vectorizer.transform(TRAIN_TEXTS))[:, 1]
    assert len(probs) == len(TRAIN_TEXTS)
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert all(not math.isnan(p) and math.isfinite(p) for p in probs)


@pytest.fixture
def trained_pair():
    config = TfidfConfig(min_df=1)
    lr_config = LogisticRegressionConfig(random_state=42)
    vectorizer = fit_vectorizer(TRAIN_TEXTS, config)
    model = fit_model(vectorizer.transform(TRAIN_TEXTS), TRAIN_LABELS, lr_config)
    return vectorizer, model
