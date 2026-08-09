from pathlib import Path

from PIL import Image

from safelens.data.multimodal.preprocessing.leakage import check_leakage


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=color).save(path, format="JPEG")


def test_clean_splits_have_no_leakage(tmp_path: Path, make_example_fn):
    _write_image(tmp_path / "a.jpg", (1, 1, 1))
    _write_image(tmp_path / "b.jpg", (2, 2, 2))
    splits = {
        "train": [make_example_fn("id1", "train text", "a.jpg", split="train")],
        "dev": [make_example_fn("id2", "dev text", "b.jpg", split="dev")],
    }
    report = check_leakage(splits, tmp_path)
    assert report.is_clean


def test_duplicate_text_across_splits_detected(tmp_path: Path, make_example_fn):
    _write_image(tmp_path / "a.jpg", (1, 1, 1))
    _write_image(tmp_path / "b.jpg", (2, 2, 2))
    splits = {
        "train": [make_example_fn("id1", "shared caption", "a.jpg", split="train")],
        "test": [make_example_fn("id2", "shared caption", "b.jpg", split="test")],
    }
    report = check_leakage(splits, tmp_path)
    assert not report.is_clean
    assert "train<->test" in report.text_overlaps


def test_duplicate_image_across_splits_detected(tmp_path: Path, make_example_fn):
    _write_image(tmp_path / "a.jpg", (5, 5, 5))
    _write_image(tmp_path / "b.jpg", (5, 5, 5))
    splits = {
        "train": [make_example_fn("id1", "text a", "a.jpg", split="train")],
        "dev": [make_example_fn("id2", "text b", "b.jpg", split="dev")],
    }
    report = check_leakage(splits, tmp_path)
    assert not report.is_clean
    assert "train<->dev" in report.image_hash_overlaps


def test_duplicate_id_across_splits_detected(tmp_path: Path, make_example_fn):
    _write_image(tmp_path / "a.jpg", (1, 1, 1))
    splits = {
        "train": [make_example_fn("shared_id", "text a", "a.jpg", split="train")],
        "dev": [make_example_fn("shared_id", "text b", "a.jpg", split="dev")],
    }
    report = check_leakage(splits, tmp_path)
    assert not report.is_clean
    assert "train<->dev" in report.id_overlaps
