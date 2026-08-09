import pytest
from pydantic import ValidationError

from safelens.data.schema import LabelScores, ModerationExample


def test_valid_example_passes(make_example_fn):
    ex = make_example_fn("This is a fine comment.", toxicity=0.1)
    assert ex.text == "This is a fine comment."


def test_empty_text_rejected(make_example_fn):
    with pytest.raises(ValidationError):
        make_example_fn("   ")


def test_label_out_of_range_rejected():
    with pytest.raises(ValidationError):
        LabelScores(
            toxicity=1.5,
            severe_toxicity=0.0,
            obscene=0.0,
            threat=0.0,
            insult=0.0,
            identity_attack=0.0,
            sexual_explicit=0.0,
        )


def test_missing_required_field_rejected():
    with pytest.raises(ValidationError):
        ModerationExample(
            content_id="abc",
            text="hello",
            labels=LabelScores(
                toxicity=0.0,
                severe_toxicity=0.0,
                obscene=0.0,
                threat=0.0,
                insult=0.0,
                identity_attack=0.0,
                sexual_explicit=0.0,
            ),
            dataset_version="v1",
            # source omitted
        )
