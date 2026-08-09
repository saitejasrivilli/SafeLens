from safelens.models.text.deberta.error_analysis import find_error_examples

CONTENT_IDS = ["c1", "c2", "c3", "c4"]
TEXTS = ["this is fine", "you are terrible and awful", "great job today", "i will hurt you"]
Y_TRUE = [0, 1, 0, 1]
Y_PROB = [0.9, 0.1, 0.05, 0.05]  # c1: false positive, c2 & c4: false negatives


def test_finds_false_positives_and_negatives():
    report = find_error_examples(CONTENT_IDS, TEXTS, Y_TRUE, Y_PROB, threshold=0.5)
    assert report["total_false_positives"] == 1
    assert report["total_false_negatives"] == 2


def test_high_confidence_false_positive_is_c1():
    report = find_error_examples(CONTENT_IDS, TEXTS, Y_TRUE, Y_PROB, threshold=0.5)
    fps = report["high_confidence_false_positives"]
    assert len(fps) == 1
    assert fps[0]["content_id"] == "c1"


def test_high_confidence_false_negatives_sorted_by_lowest_probability():
    report = find_error_examples(CONTENT_IDS, TEXTS, Y_TRUE, Y_PROB, threshold=0.5)
    fns = report["high_confidence_false_negatives"]
    assert len(fns) == 2
    assert fns[0]["predicted_probability"] <= fns[1]["predicted_probability"]


def test_text_is_truncated():
    long_text = "x" * 500
    report = find_error_examples(["c1"], [long_text], [0], [0.9], threshold=0.5)
    excerpt = report["high_confidence_false_positives"][0]["text_excerpt"]
    assert len(excerpt) <= 103  # MAX_CHARS + "..."


def test_no_errors_when_all_correct():
    report = find_error_examples(["c1", "c2"], ["a", "b"], [0, 1], [0.1, 0.9], threshold=0.5)
    assert report["total_false_positives"] == 0
    assert report["total_false_negatives"] == 0
    assert report["high_confidence_false_positives"] == []
    assert report["high_confidence_false_negatives"] == []
