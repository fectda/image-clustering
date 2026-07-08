"""File system scanning for images."""

import logging
import random
import sys
from pathlib import Path

log = logging.getLogger("cluster")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def scan_images(input_dir: str) -> list[Path]:
    """Recursively scan input_dir for image files."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        log.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    files = sorted(
        p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        log.error("No images found (jpg/jpeg/png/webp) in: %s", input_dir)
        sys.exit(1)
    log.info("Found %d images", len(files))
    return files


def sample_preview(files: list[Path], n: int = 10) -> list[Path]:
    return files if len(files) <= n else random.sample(files, n)
