"""Tests for scan_images recursive flag."""

from pathlib import Path

import pytest

from photo_organizer.scanner import ScanError, scan_images


def _make_fake_jpeg(path: Path):
    """Create minimal valid JPEG (JFIF header)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        bytes(
            [
                0xFF,
                0xD8,
                0xFF,
                0xE0,
                0x00,
                0x10,
                0x4A,
                0x46,
                0x49,
                0x46,
                0x00,
                0x01,
                0x01,
                0x00,
                0x00,
                0x01,
                0x00,
                0x01,
                0x00,
                0x00,
                0xFF,
                0xD9,
            ]
        )
    )


class TestScanImagesRecursive:
    """scan_images() with recursive=True (default)."""

    def test_recursive_finds_root_and_subdir_images(self, tmp_path):
        """Default recursive=True finds images in root and subdirectories."""
        _make_fake_jpeg(tmp_path / "root.jpg")
        _make_fake_jpeg(tmp_path / "subdir" / "nested.jpg")
        _make_fake_jpeg(tmp_path / "subdir" / "deep" / "deep.jpg")

        result = scan_images(str(tmp_path))

        names = [p.name for p in result]
        assert "root.jpg" in names
        assert "nested.jpg" in names
        assert "deep.jpg" in names
        assert len(result) == 3


class TestScanImagesNonRecursive:
    """scan_images() with recursive=False."""

    def test_non_recursive_finds_only_root_images(self, tmp_path):
        """recursive=False finds only root-level images."""
        _make_fake_jpeg(tmp_path / "root.jpg")
        _make_fake_jpeg(tmp_path / "subdir" / "nested.jpg")
        _make_fake_jpeg(tmp_path / "subdir" / "deep" / "deep.jpg")

        result = scan_images(str(tmp_path), recursive=False)

        names = [p.name for p in result]
        assert "root.jpg" in names
        assert "nested.jpg" not in names
        assert "deep.jpg" not in names
        assert len(result) == 1

    def test_non_recursive_with_no_images_fails(self, tmp_path):
        """recursive=False with no root images → ScanError."""
        _make_fake_jpeg(tmp_path / "subdir" / "nested.jpg")

        with pytest.raises(ScanError):
            scan_images(str(tmp_path), recursive=False)

    def test_non_recursive_empty_dir_fails(self, tmp_path):
        """recursive=False with empty directory → ScanError."""
        with pytest.raises(ScanError):
            scan_images(str(tmp_path), recursive=False)
