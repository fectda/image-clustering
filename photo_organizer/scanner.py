"""File system scanning for images."""

import logging
import random
from pathlib import Path

log = logging.getLogger("cluster")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class ScanError(Exception):
    """Raised when image scanning fails (directory missing or no images found)."""


def scan_images(input_dir: str, recursive: bool = True) -> list[Path]:
    """Scan input_dir for image files.

    When recursive=True (default), scans all subdirectories.
    When recursive=False, only scans the root directory.

    Raises
    ------
    ScanError
        If input_dir does not exist or no images are found.
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise ScanError(f"Input directory does not exist: {input_dir}")

    files = []
    glob_method = input_path.rglob if recursive else input_path.glob
    for p in glob_method("*"):
        try:
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                files.append(p)
        except (PermissionError, OSError):
            log.warning("Skipping unreadable file: %s", p)
            continue
    files = sorted(files)
    if not files:
        raise ScanError(f"No images found (jpg/jpeg/png/webp) in: {input_dir}")
    log.info("Found %d images", len(files))
    return files


def sample_preview(files: list[Path], n: int = 10) -> list[Path]:
    return files if len(files) <= n else random.sample(files, n)
