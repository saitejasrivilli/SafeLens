"""Data contract for Prop2Hate-Meme examples.

Source: https://huggingface.co/datasets/QCRI/Prop2Hate-Meme (CC-BY-NC-SA-4.0).
This is a SEPARATE experimental track from the Phase 2-4 civil_comments
pipeline -- nothing here is combined with or substitutes for that data.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

HATE_LABEL_NAMES = ["not-hateful", "hateful"]
PROP_LABEL_NAMES = ["not_propaganda", "propaganda"]
HATE_FINE_GRAINED_LABEL_NAMES = [
    "sarcasm",
    "humor",
    "inciting_violence",
    "mocking",
    "other",
    "exclusion",
    "dehumanizing",
    "contempt",
    "inferiority",
    "slurs",
]


class MultimodalExample(BaseModel):
    example_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    image_path: str = Field(
        min_length=1
    )  # path to the extracted image file, relative to processed dir
    source_img_path: str = Field(
        min_length=1
    )  # original dataset img_path field, preserved for traceability
    hate_label: int = Field(ge=0, le=1)
    prop_label: int = Field(ge=0, le=1)
    hate_fine_grained_label: int = Field(ge=0, le=9)
    split: str
    dataset_version: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text is empty after stripping whitespace")
        return v

    @field_validator("split")
    @classmethod
    def split_must_be_known(cls, v: str) -> str:
        if v not in {"train", "dev", "test"}:
            raise ValueError(f"unexpected split value: {v!r}")
        return v

    @property
    def hate_label_name(self) -> str:
        return HATE_LABEL_NAMES[self.hate_label]

    @property
    def prop_label_name(self) -> str:
        return PROP_LABEL_NAMES[self.prop_label]

    @property
    def hate_fine_grained_label_name(self) -> str:
        return HATE_FINE_GRAINED_LABEL_NAMES[self.hate_fine_grained_label]
