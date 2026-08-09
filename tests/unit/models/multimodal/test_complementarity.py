from safelens.models.multimodal.complementarity import analyze_complementarity


def test_complementarity_categorizes_correctly():
    # id1: text wrong, image wrong, fusion correct -> "both wrong, fusion right"
    # id2: text correct, image wrong, fusion wrong
    # id3: image correct, text wrong, fusion wrong
    # id4: all correct
    # id5: all wrong
    ids = ["id1", "id2", "id3", "id4", "id5"]
    y_true = [1, 1, 1, 0, 1]
    text_probs = [0.1, 0.9, 0.1, 0.1, 0.1]  # wrong, correct, wrong, correct, wrong
    image_probs = [0.1, 0.1, 0.9, 0.1, 0.1]  # wrong, wrong, correct, correct, wrong
    fusion_probs = [0.9, 0.1, 0.1, 0.1, 0.1]  # correct, wrong, wrong, correct, wrong

    result = analyze_complementarity(
        ids, y_true, text_probs, image_probs, fusion_probs, threshold=0.5
    )

    assert result["counts"]["text_wrong_image_wrong_fusion_correct"] == 1
    assert result["example_ids"]["text_wrong_image_wrong_fusion_correct"] == ["id1"]
    assert result["counts"]["text_correct_image_wrong_fusion_wrong"] == 1
    assert result["example_ids"]["text_correct_image_wrong_fusion_wrong"] == ["id2"]
    assert result["counts"]["image_correct_text_wrong_fusion_wrong"] == 1
    assert result["example_ids"]["image_correct_text_wrong_fusion_wrong"] == ["id3"]
    assert result["counts"]["all_three_correct"] == 1
    assert result["counts"]["all_three_wrong"] == 1
    assert result["n"] == 5


def test_complementarity_accuracy_computation():
    ids = ["id1", "id2"]
    y_true = [1, 0]
    text_probs = [0.9, 0.9]  # correct, wrong
    image_probs = [0.1, 0.1]  # wrong, correct
    fusion_probs = [0.9, 0.1]  # correct, correct

    result = analyze_complementarity(ids, y_true, text_probs, image_probs, fusion_probs)
    assert result["accuracy_at_threshold"]["text_only"] == 0.5
    assert result["accuracy_at_threshold"]["image_only"] == 0.5
    assert result["accuracy_at_threshold"]["multimodal"] == 1.0


def test_complementarity_no_examples_no_crash():
    result = analyze_complementarity([], [], [], [], [])
    assert result["n"] == 0
    assert result["counts"]["all_three_correct"] == 0
