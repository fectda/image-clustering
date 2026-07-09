"""Tests for organize_photos() new parameters: recursive, no_gallery, output_mode."""

from pathlib import Path
from unittest.mock import patch

import numpy as np

from photo_organizer.main import organize_photos


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


def _mock_pipeline(tmp_path):
    """Set up fake images and return mock config for GPU-heavy pipeline."""
    src_dir = tmp_path / "src"
    img1 = src_dir / "photo1.jpg"
    img2 = src_dir / "photo2.jpg"
    _make_fake_jpeg(img1)
    _make_fake_jpeg(img2)

    # Mock returns: model, processor, backend, device
    mock_model = "mock_model"
    mock_processor = "mock_processor"
    mock_backend = "cpu"
    mock_device = "cpu"

    # Mock embeddings
    mock_embeddings = np.random.rand(2, 128).astype(np.float32)

    # Mock prefixes: both images in cluster "c0_"
    mock_prefixes = {0: "c0_", 1: "c0_"}

    return {
        "images": [img1, img2],
        "mock_model": (mock_model, mock_processor, mock_backend, mock_device),
        "mock_embeddings": (mock_embeddings, [img1, img2]),
        "mock_prefixes": mock_prefixes,
    }


class TestOrganizePhotosNoGallery:
    """organize_photos() with no_gallery=True skips gallery generation."""

    def test_no_gallery_skips_gallery(self, tmp_path):
        """no_gallery=True: files moved but no index.html created."""
        ctx = _mock_pipeline(tmp_path)
        output_dir = tmp_path / "out"

        with (
            patch("photo_organizer.main.load_model", return_value=ctx["mock_model"]),
            patch("photo_organizer.main.extract_embeddings", return_value=ctx["mock_embeddings"]),
            patch("photo_organizer.main.recursive_cluster", return_value=ctx["mock_prefixes"]),
            patch("photo_organizer.main.classify_noise_with_clip"),
            patch("photo_organizer.main.generate_flat_gallery") as mock_gallery,
        ):
            organize_photos(
                str(tmp_path / "src"),
                str(output_dir),
                no_gallery=True,
            )

        # Gallery should NOT be called
        mock_gallery.assert_not_called()
        # Files should be moved
        assert (output_dir / "c0_photo1.jpg").exists()
        assert (output_dir / "c0_photo2.jpg").exists()

    def test_default_generates_gallery(self, tmp_path):
        """Default (no_gallery=False): gallery IS generated."""
        ctx = _mock_pipeline(tmp_path)
        output_dir = tmp_path / "out"

        with (
            patch("photo_organizer.main.load_model", return_value=ctx["mock_model"]),
            patch("photo_organizer.main.extract_embeddings", return_value=ctx["mock_embeddings"]),
            patch("photo_organizer.main.recursive_cluster", return_value=ctx["mock_prefixes"]),
            patch("photo_organizer.main.classify_noise_with_clip"),
            patch("photo_organizer.main.generate_flat_gallery") as mock_gallery,
        ):
            organize_photos(
                str(tmp_path / "src"),
                str(output_dir),
            )

        mock_gallery.assert_called_once()


class TestOrganizePhotosOutputModeFolders:
    """organize_photos() with output_mode='folders' uses folders_export."""

    def test_folders_mode_creates_dirs(self, tmp_path):
        """output_mode='folders': physical subdirs created, no prefix in filenames."""
        ctx = _mock_pipeline(tmp_path)
        output_dir = tmp_path / "out"

        with (
            patch("photo_organizer.main.load_model", return_value=ctx["mock_model"]),
            patch("photo_organizer.main.extract_embeddings", return_value=ctx["mock_embeddings"]),
            patch("photo_organizer.main.recursive_cluster", return_value=ctx["mock_prefixes"]),
            patch("photo_organizer.main.classify_noise_with_clip"),
            patch("photo_organizer.main.generate_flat_gallery"),
        ):
            organize_photos(
                str(tmp_path / "src"),
                str(output_dir),
                output_mode="folders",
            )

        # Folders mode: c0/photo1.jpg, not c0_photo1.jpg
        assert (output_dir / "c0" / "photo1.jpg").exists()
        assert (output_dir / "c0" / "photo2.jpg").exists()
        # Flat-prefixed files should NOT exist
        assert not (output_dir / "c0_photo1.jpg").exists()

    def test_flat_mode_default(self, tmp_path):
        """Default output_mode='flat': flat prefixed files."""
        ctx = _mock_pipeline(tmp_path)
        output_dir = tmp_path / "out"

        with (
            patch("photo_organizer.main.load_model", return_value=ctx["mock_model"]),
            patch("photo_organizer.main.extract_embeddings", return_value=ctx["mock_embeddings"]),
            patch("photo_organizer.main.recursive_cluster", return_value=ctx["mock_prefixes"]),
            patch("photo_organizer.main.classify_noise_with_clip"),
            patch("photo_organizer.main.generate_flat_gallery"),
        ):
            organize_photos(
                str(tmp_path / "src"),
                str(output_dir),
                output_mode="flat",
            )

        assert (output_dir / "c0_photo1.jpg").exists()
        assert not (output_dir / "c0" / "photo1.jpg").exists()


class TestOrganizePhotosRecursive:
    """organize_photos() with recursive=False only scans root."""

    def test_non_recursive_skips_subdirs(self, tmp_path):
        """recursive=False: only root images processed."""
        src_dir = tmp_path / "src"
        root_img = src_dir / "root.jpg"
        subdir_img = src_dir / "subdir" / "nested.jpg"
        _make_fake_jpeg(root_img)
        _make_fake_jpeg(subdir_img)

        output_dir = tmp_path / "out"

        # Only root image should be in the pipeline
        mock_embeddings = np.random.rand(1, 128).astype(np.float32)
        mock_prefixes = {0: "c0_"}

        with (
            patch("photo_organizer.main.load_model", return_value=("m", "p", "cpu", "cpu")),
            patch(
                "photo_organizer.main.extract_embeddings",
                return_value=(mock_embeddings, [root_img]),
            ),
            patch("photo_organizer.main.recursive_cluster", return_value=mock_prefixes),
            patch("photo_organizer.main.classify_noise_with_clip"),
            patch("photo_organizer.main.generate_flat_gallery"),
        ):
            organize_photos(
                str(src_dir),
                str(output_dir),
                recursive=False,
            )

        # Only root image exported
        assert (output_dir / "c0_root.jpg").exists()
        # Nested image NOT processed (still in source)
        assert subdir_img.exists()

    def test_recursive_default_processes_all(self, tmp_path):
        """Default recursive=True: all images processed."""
        src_dir = tmp_path / "src"
        root_img = src_dir / "root.jpg"
        subdir_img = src_dir / "subdir" / "nested.jpg"
        _make_fake_jpeg(root_img)
        _make_fake_jpeg(subdir_img)

        output_dir = tmp_path / "out"

        mock_embeddings = np.random.rand(2, 128).astype(np.float32)
        mock_prefixes = {0: "c0_", 1: "c0_"}

        with (
            patch("photo_organizer.main.load_model", return_value=("m", "p", "cpu", "cpu")),
            patch(
                "photo_organizer.main.extract_embeddings",
                return_value=(mock_embeddings, [root_img, subdir_img]),
            ),
            patch("photo_organizer.main.recursive_cluster", return_value=mock_prefixes),
            patch("photo_organizer.main.classify_noise_with_clip"),
            patch("photo_organizer.main.generate_flat_gallery"),
        ):
            organize_photos(
                str(src_dir),
                str(output_dir),
            )

        # Both images exported
        assert (output_dir / "c0_root.jpg").exists()
        assert (output_dir / "c0_nested.jpg").exists()
