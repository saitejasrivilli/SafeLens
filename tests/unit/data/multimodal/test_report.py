import json
from pathlib import Path

from PIL import Image

from safelens.data.multimodal.validation.report import build_validation_report, load_valid_examples

GOOD_ROW = {
    "example_id": "id1",
    "text": "نص عربي صحيح",
    "image_path": "images/train/a.jpg",
    "source_img_path": "./a.jpg",
    "hate_label": 1,
    "prop_label": 0,
    "hate_fine_grained_label": 2,
    "split": "train",
    "dataset_version": "QCRI/Prop2Hate-Meme@test",
}


def _write_split(processed_dir: Path, split: str, rows: list[dict]) -> None:
    (processed_dir / f"{split}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    )


def _write_valid_image(processed_dir: Path, rel_path: str) -> None:
    path = processed_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(1, 2, 3)).save(path, format="JPEG")


def test_build_validation_report_all_valid(tmp_path: Path):
    _write_valid_image(tmp_path, GOOD_ROW["image_path"])
    _write_split(tmp_path, "train", [GOOD_ROW])
    _write_split(tmp_path, "dev", [])
    _write_split(tmp_path, "test", [])

    report = build_validation_report(tmp_path)
    assert report["splits"]["train"]["total_rows"] == 1
    assert report["splits"]["train"]["valid_rows"] == 1
    assert report["splits"]["train"]["malformed_rows"] == 0
    assert report["splits"]["train"]["image_issues"] == 0
    assert report["image_validation_summary"]["valid"] == 1
    assert report["arabic_text_fraction_overall"] == 1.0


def test_build_validation_report_flags_malformed_row(tmp_path: Path):
    bad_row = dict(GOOD_ROW, hate_label=5)  # out of range
    _write_valid_image(tmp_path, GOOD_ROW["image_path"])
    _write_split(tmp_path, "train", [GOOD_ROW, bad_row])
    _write_split(tmp_path, "dev", [])
    _write_split(tmp_path, "test", [])

    report = build_validation_report(tmp_path)
    assert report["splits"]["train"]["malformed_rows"] == 1
    assert report["splits"]["train"]["valid_rows"] == 1


def test_build_validation_report_flags_missing_image(tmp_path: Path):
    row = dict(GOOD_ROW, image_path="images/train/missing.jpg")
    _write_split(tmp_path, "train", [row])
    _write_split(tmp_path, "dev", [])
    _write_split(tmp_path, "test", [])

    report = build_validation_report(tmp_path)
    assert report["image_validation_summary"]["missing"] == 1
    assert report["splits"]["train"]["image_issues"] == 1


def test_load_valid_examples_skips_malformed(tmp_path: Path):
    bad_row = dict(GOOD_ROW, hate_label=5)
    _write_valid_image(tmp_path, GOOD_ROW["image_path"])
    _write_split(tmp_path, "train", [GOOD_ROW, bad_row])
    _write_split(tmp_path, "dev", [])
    _write_split(tmp_path, "test", [])

    by_split = load_valid_examples(tmp_path)
    assert len(by_split["train"]) == 1
    assert by_split["dev"] == []
