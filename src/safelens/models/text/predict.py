"""Policy-independent prediction output. No ALLOW/REVIEW/BLOCK decision here
— that's the moderation policy engine (Phase 10)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    content_id: str
    toxicity_probability: float
    threshold: float
    predicted_label: int
    model_name: str
    model_version: str


def predict_batch(
    model,
    vectorizer,
    content_ids: list[str],
    texts: list[str],
    threshold: float,
    model_name: str,
    model_version: str,
) -> list[Prediction]:
    X = vectorizer.transform(texts)
    probs = model.predict_proba(X)[:, 1]
    return [
        Prediction(
            content_id=cid,
            toxicity_probability=float(p),
            threshold=threshold,
            predicted_label=int(p >= threshold),
            model_name=model_name,
            model_version=model_version,
        )
        for cid, p in zip(content_ids, probs, strict=True)
    ]
