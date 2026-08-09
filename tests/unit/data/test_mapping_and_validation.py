import pytest
from pydantic import ValidationError

from safelens.data.mapping import MalformedRowError, civil_comments_row_to_example
from safelens.data.validation.report import build_validation_report

GOOD_ROW = {
    "row_idx": 0,
    "text": "Have a great day!",
    "toxicity": 0.0,
    "severe_toxicity": 0.0,
    "obscene": 0.0,
    "threat": 0.0,
    "insult": 0.0,
    "identity_attack": 0.0,
    "sexual_explicit": 0.0,
}


def test_good_row_maps_cleanly():
    ex = civil_comments_row_to_example(GOOD_ROW, dataset_version="v1")
    assert ex.text == GOOD_ROW["text"]
    assert ex.labels.toxicity == 0.0


def test_missing_field_raises_malformed():
    bad = {k: v for k, v in GOOD_ROW.items() if k != "toxicity"}
    with pytest.raises(MalformedRowError):
        civil_comments_row_to_example(bad, dataset_version="v1")


def test_non_numeric_label_raises_malformed():
    bad = dict(GOOD_ROW, toxicity="not-a-number")
    with pytest.raises(MalformedRowError):
        civil_comments_row_to_example(bad, dataset_version="v1")


def test_out_of_range_label_raises_validation_error():
    bad = dict(GOOD_ROW, toxicity=2.0)
    with pytest.raises(ValidationError):
        civil_comments_row_to_example(bad, dataset_version="v1")


def test_empty_text_raises_validation_error():
    bad = dict(GOOD_ROW, text="   ")
    with pytest.raises(ValidationError):
        civil_comments_row_to_example(bad, dataset_version="v1")


def test_validation_report_counts():
    rows = [
        GOOD_ROW,
        dict(GOOD_ROW, row_idx=1, text="Another fine comment."),
        {k: v for k, v in GOOD_ROW.items() if k != "toxicity"} | {"row_idx": 2},  # malformed
        dict(GOOD_ROW, row_idx=3, toxicity=2.0),  # invalid range
        dict(GOOD_ROW, row_idx=4, text=GOOD_ROW["text"]),  # exact duplicate content
    ]
    report = build_validation_report(rows, dataset_version="v1")
    assert report["total_rows"] == 5
    assert report["valid_rows"] == 3  # two good + one duplicate-content (still schema-valid)
    assert report["malformed_rows"] == 1
    assert report["invalid_rows"] == 1
    # content_id is derived from text, so exact-text duplicates surface as
    # duplicate-ID removals, not separately as duplicate_content_found.
    assert report["duplicate_ids_found"] == 1
