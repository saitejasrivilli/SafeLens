import pytest
from pydantic import ValidationError

from safelens.data.multimodal.schema import MultimodalExample


def test_valid_example_passes(make_example_fn):
    ex = make_example_fn("id1", "نص عربي", "images/train/a.jpg", hate_label=1)
    assert ex.hate_label_name == "hateful"
    assert ex.prop_label_name == "not_propaganda"
    assert ex.hate_fine_grained_label_name == "humor"


def test_empty_text_rejected(make_example_fn):
    with pytest.raises(ValidationError):
        make_example_fn("id1", "   ", "images/train/a.jpg")


def test_invalid_hate_label_rejected(make_example_fn):
    with pytest.raises(ValidationError):
        make_example_fn("id1", "text", "images/train/a.jpg", hate_label=2)


def test_invalid_fine_grained_label_rejected(make_example_fn):
    with pytest.raises(ValidationError):
        make_example_fn("id1", "text", "images/train/a.jpg", hate_fine_grained_label=99)


def test_unknown_split_rejected(make_example_fn):
    with pytest.raises(ValidationError):
        make_example_fn("id1", "text", "images/train/a.jpg", split="validation")


def test_missing_required_field_rejected():
    with pytest.raises(ValidationError):
        MultimodalExample(
            example_id="id1",
            text="text",
            image_path="images/train/a.jpg",
            hate_label=0,
            prop_label=0,
            hate_fine_grained_label=1,
            split="train",
            # dataset_version omitted
        )
