from pathlib import Path

from PIL import Image

from safelens.data.multimodal.preprocessing.dedup import find_duplicates


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=color).save(path, format="JPEG")


def test_no_duplicates(tmp_path: Path, make_example_fn):
    _write_image(tmp_path / "a.jpg", (1, 2, 3))
    _write_image(tmp_path / "b.jpg", (4, 5, 6))
    examples = [
        make_example_fn("id1", "text one", "a.jpg"),
        make_example_fn("id2", "text two", "b.jpg"),
    ]
    report = find_duplicates(examples, tmp_path)
    assert report.duplicate_ids == []
    assert report.duplicate_texts == []
    assert report.duplicate_image_hashes == []
    assert report.duplicate_pairs == []


def test_duplicate_text_across_different_images(tmp_path: Path, make_example_fn):
    _write_image(tmp_path / "a.jpg", (1, 2, 3))
    _write_image(tmp_path / "b.jpg", (4, 5, 6))
    examples = [
        make_example_fn("id1", "same caption", "a.jpg"),
        make_example_fn("id2", "same caption", "b.jpg"),
    ]
    report = find_duplicates(examples, tmp_path)
    assert report.duplicate_texts == ["same caption"]
    assert report.duplicate_image_hashes == []  # different images, not flagged
    assert report.duplicate_pairs == []  # text+image pair differs


def test_duplicate_image_hash_detected(tmp_path: Path, make_example_fn):
    _write_image(tmp_path / "a.jpg", (7, 7, 7))
    _write_image(tmp_path / "b.jpg", (7, 7, 7))  # identical pixels -> identical bytes
    examples = [
        make_example_fn("id1", "text one", "a.jpg"),
        make_example_fn("id2", "text two", "b.jpg"),
    ]
    report = find_duplicates(examples, tmp_path)
    assert len(report.duplicate_image_hashes) == 1


def test_duplicate_full_pair_detected(tmp_path: Path, make_example_fn):
    _write_image(tmp_path / "a.jpg", (9, 9, 9))
    _write_image(tmp_path / "b.jpg", (9, 9, 9))
    examples = [
        make_example_fn("id1", "identical caption", "a.jpg"),
        make_example_fn("id2", "identical caption", "b.jpg"),
    ]
    report = find_duplicates(examples, tmp_path)
    assert len(report.duplicate_pairs) == 1
