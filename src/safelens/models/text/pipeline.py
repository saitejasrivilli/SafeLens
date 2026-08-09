"""TF-IDF + Logistic Regression fitting. The vectorizer is fit ONLY on
training text — callers must never pass validation/test text to fit()."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from safelens.models.text.config import LogisticRegressionConfig, TfidfConfig


def fit_vectorizer(train_texts: list[str], config: TfidfConfig) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
        max_features=config.max_features,
        sublinear_tf=config.sublinear_tf,
    )
    vectorizer.fit(train_texts)
    return vectorizer


def fit_model(
    X_train,
    y_train: list[int],
    config: LogisticRegressionConfig,  # noqa: N803
) -> LogisticRegression:
    model = LogisticRegression(
        C=config.C,
        class_weight=config.class_weight,
        max_iter=config.max_iter,
        random_state=config.random_state,
    )
    model.fit(X_train, y_train)
    return model
