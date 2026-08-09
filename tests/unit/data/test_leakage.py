from safelens.data.preprocessing.leakage import check_leakage


def test_clean_splits_have_no_leakage(make_example_fn):
    splits = {
        "train": [make_example_fn("train comment one"), make_example_fn("train comment two")],
        "validation": [make_example_fn("val comment one")],
        "test": [make_example_fn("test comment one")],
    }
    report = check_leakage(splits)
    assert report.is_clean
    assert report.id_overlaps == {}
    assert report.normalized_text_overlaps == {}


def test_duplicate_id_across_splits_detected(make_example_fn):
    shared = make_example_fn("shared comment")
    splits = {
        "train": [shared],
        "validation": [shared],
    }
    report = check_leakage(splits)
    assert not report.is_clean
    assert "train<->validation" in report.id_overlaps


def test_normalized_text_overlap_across_splits_detected(make_example_fn):
    splits = {
        "train": [make_example_fn("Hello, World!!")],
        "test": [make_example_fn("hello world")],
    }
    report = check_leakage(splits)
    assert not report.is_clean
    assert "train<->test" in report.normalized_text_overlaps


def test_distinct_symbol_only_text_not_flagged_as_leakage(make_example_fn):
    """Different raw text that both normalize to "" (pure punctuation/emoji)
    must not be treated as a duplicate across splits."""
    splits = {
        "train": [make_example_fn("!!!")],
        "test": [make_example_fn("???")],
    }
    report = check_leakage(splits)
    assert report.is_clean
