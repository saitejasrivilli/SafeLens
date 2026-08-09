import math
from pathlib import Path

from safelens.models.text.artifacts import load_artifacts, save_artifacts
from safelens.models.text.config import BaselineConfig, TfidfConfig
from safelens.models.text.pipeline import fit_model, fit_vectorizer
from safelens.models.text.predict import Prediction, predict_batch

TRAIN_TEXTS = ["nice comment here", "you are terrible and awful", "have a good one"]
TRAIN_LABELS = [0, 1, 0]


def _trained():
    vectorizer = fit_vectorizer(TRAIN_TEXTS, TfidfConfig(min_df=1))
    lr_config = BaselineConfig().logistic_regression
    model = fit_model(vectorizer.transform(TRAIN_TEXTS), TRAIN_LABELS, lr_config)
    return vectorizer, model


def test_save_and_load_artifacts_round_trip(tmp_path: Path):
    vectorizer, model = _trained()
    config = BaselineConfig(model_version="test-v1")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")

    out_dir = save_artifacts(
        model_version="test-v1",
        vectorizer=vectorizer,
        model=model,
        config=config,
        dataset_version="fake@0-3",
        dataset_manifest_path=manifest_path,
        root=tmp_path / "models",
    )
    assert (out_dir / "vectorizer.joblib").exists()
    assert (out_dir / "model.joblib").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "metadata.json").exists()

    loaded_vectorizer, loaded_model, loaded_config = load_artifacts(
        "test-v1", root=tmp_path / "models"
    )
    assert loaded_config.model_version == "test-v1"

    original_probs = model.predict_proba(vectorizer.transform(TRAIN_TEXTS))[:, 1]
    loaded_probs = loaded_model.predict_proba(loaded_vectorizer.transform(TRAIN_TEXTS))[:, 1]
    assert list(original_probs) == list(loaded_probs)


def test_predict_batch_shape_and_fields():
    vectorizer, model = _trained()
    content_ids = ["c1", "c2", "c3"]
    predictions = predict_batch(
        model,
        vectorizer,
        content_ids,
        TRAIN_TEXTS,
        threshold=0.5,
        model_name="baseline-tfidf-logreg",
        model_version="test-v1",
    )
    assert len(predictions) == len(TRAIN_TEXTS)
    for pred, cid in zip(predictions, content_ids, strict=True):
        assert isinstance(pred, Prediction)
        assert pred.content_id == cid
        assert 0.0 <= pred.toxicity_probability <= 1.0
        assert not math.isnan(pred.toxicity_probability)
        assert pred.predicted_label in (0, 1)
        assert pred.threshold == 0.5
        assert pred.model_name == "baseline-tfidf-logreg"
        assert pred.model_version == "test-v1"
