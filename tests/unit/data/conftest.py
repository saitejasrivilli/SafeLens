import pytest

from safelens.data.preprocessing.normalize import content_id_for
from safelens.data.schema import LabelScores, ModerationExample


def make_example(text: str, toxicity: float = 0.0, **overrides) -> ModerationExample:
    defaults = dict(
        content_id=content_id_for(text),
        text=text,
        image_ref=None,
        labels=LabelScores(
            toxicity=toxicity,
            severe_toxicity=0.0,
            obscene=0.0,
            threat=0.0,
            insult=0.0,
            identity_attack=0.0,
            sexual_explicit=0.0,
        ),
        source="google/civil_comments",
        timestamp=None,
        dataset_version="test-fixture-v1",
    )
    defaults.update(overrides)
    return ModerationExample(**defaults)


@pytest.fixture
def make_example_fn():
    return make_example
