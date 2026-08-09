"""Data contract for SafeLens training examples.

civil_comments provides continuous per-attribute toxicity scores (fraction of
annotators who flagged that attribute), not binary labels — the schema
reflects that directly instead of forcing a binary policy label at ingest
time. Binarization/thresholding is a moderation-policy decision (Phase 10),
not a data-contract decision.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class LabelScores(BaseModel):
    toxicity: float = Field(ge=0.0, le=1.0)
    severe_toxicity: float = Field(ge=0.0, le=1.0)
    obscene: float = Field(ge=0.0, le=1.0)
    threat: float = Field(ge=0.0, le=1.0)
    insult: float = Field(ge=0.0, le=1.0)
    identity_attack: float = Field(ge=0.0, le=1.0)
    sexual_explicit: float = Field(ge=0.0, le=1.0)


class ModerationExample(BaseModel):
    content_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    image_ref: str | None = None
    labels: LabelScores
    source: str = Field(min_length=1)
    timestamp: str | None = None
    dataset_version: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text is empty after stripping whitespace")
        return v
