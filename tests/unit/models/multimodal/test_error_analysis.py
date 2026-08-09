from safelens.models.multimodal.error_analysis import find_error_examples

IDS = ["c1", "c2", "c3", "c4"]
TEXTS = ["نص واحد", "نص اثنين", "نص ثلاثة", "نص اربعة"]
IMAGE_PATHS = ["images/a.jpg", "images/b.jpg", "images/c.jpg", "images/d.jpg"]
Y_TRUE = [0, 1, 0, 1]
Y_PROB = [0.9, 0.1, 0.05, 0.05]  # c1: FP, c2 & c4: FN


def test_finds_false_positives_and_negatives():
    report = find_error_examples(IDS, TEXTS, IMAGE_PATHS, Y_TRUE, Y_PROB, threshold=0.5)
    assert report["total_false_positives"] == 1
    assert report["total_false_negatives"] == 2


def test_error_rows_include_both_text_and_image():
    report = find_error_examples(IDS, TEXTS, IMAGE_PATHS, Y_TRUE, Y_PROB, threshold=0.5)
    for row in (
        report["high_confidence_false_positives"] + report["high_confidence_false_negatives"]
    ):
        assert "text_excerpt" in row
        assert "image_path" in row
