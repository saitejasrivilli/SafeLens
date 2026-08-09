"""Image validation: existence, decodability, dimensions, channels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageCheckResult:
    exists: bool
    decodable: bool
    width: int | None
    height: int | None
    mode: str | None
    error: str | None = None


def check_image(path: Path) -> ImageCheckResult:
    if not path.exists():
        return ImageCheckResult(
            exists=False, decodable=False, width=None, height=None, mode=None, error="missing"
        )

    try:
        from PIL import Image

        with Image.open(path) as img:
            img.verify()  # raises on structural corruption without full decode
        # verify() invalidates the file handle for further use -- reopen to
        # read actual pixel data and confirm it's genuinely decodable, not
        # just structurally parseable.
        with Image.open(path) as img:
            img.load()
            width, height = img.size
            mode = img.mode
    except Exception as exc:  # noqa: BLE001 - any decode failure means "corrupted"
        return ImageCheckResult(
            exists=True, decodable=False, width=None, height=None, mode=None, error=str(exc)
        )

    if width <= 0 or height <= 0:
        return ImageCheckResult(
            exists=True,
            decodable=False,
            width=width,
            height=height,
            mode=mode,
            error="non-positive dimensions",
        )

    return ImageCheckResult(exists=True, decodable=True, width=width, height=height, mode=mode)
