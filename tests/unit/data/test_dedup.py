from safelens.data.preprocessing.dedup import deduplicate
from safelens.data.preprocessing.normalize import content_id_for


def test_exact_content_duplicate_removed(make_example_fn):
    a = make_example_fn("hello world")
    b = make_example_fn("hello world")  # same text -> same content_id too
    result = deduplicate([a, b])
    assert len(result.unique) == 1
    assert len(result.duplicate_ids_removed) == 1


def test_duplicate_id_with_different_text_removed(make_example_fn):
    a = make_example_fn("hello world")
    b = make_example_fn("different text", content_id=a.content_id)
    result = deduplicate([a, b])
    assert len(result.unique) == 1
    assert result.duplicate_ids_removed == [a.content_id]


def test_normalized_duplicate_detected_but_not_removed(make_example_fn):
    a = make_example_fn("Hello, World!!")
    b = make_example_fn("hello world")
    result = deduplicate([a, b])
    assert len(result.unique) == 2  # not removed
    assert result.normalized_duplicate_groups == 1
    assert result.normalized_duplicate_examples == 2


def test_no_duplicates(make_example_fn):
    a = make_example_fn("first comment")
    b = make_example_fn("second comment")
    result = deduplicate([a, b])
    assert len(result.unique) == 2
    assert result.duplicate_ids_removed == []
    assert result.duplicate_content_removed == []


def test_content_id_deterministic():
    assert content_id_for("same text") == content_id_for("same text")
    assert content_id_for("text a") != content_id_for("text b")
