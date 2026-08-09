from pathlib import Path

from safelens.data.multimodal.validation.images import check_image


def test_valid_image_passes(tiny_jpeg: Path):
    result = check_image(tiny_jpeg)
    assert result.exists
    assert result.decodable
    assert result.width == 4
    assert result.height == 4
    assert result.mode == "RGB"


def test_missing_image_reported(tmp_path: Path):
    result = check_image(tmp_path / "does_not_exist.jpg")
    assert not result.exists
    assert not result.decodable
    assert result.error == "missing"


def test_corrupted_image_reported(tmp_path: Path):
    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"this is not a real jpeg file at all")
    result = check_image(path)
    assert result.exists
    assert not result.decodable
    assert result.error is not None
