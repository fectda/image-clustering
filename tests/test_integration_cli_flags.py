"""Integration tests for CLI flags — full pipeline with mocked GPU components.

Tests real scanning, and export logic on tmp_path with fake JPEG images.
GPU-heavy components (load_model, extract_embeddings, recursive_cluster) are mocked.
"""

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


def _setup_images(tmp_path, filenames, subdir=None):
    """Create fake JPEG images and return list of Paths."""
    src_dir = tmp_path / "src"
    if subdir:
        src_dir = src_dir / subdir
    paths = []
    for name in filenames:
        p = src_dir / name
        _make_fake_jpeg(p)
        paths.append(p)
    return paths


def _mock_prefixes(n, prefix="c0_"):
    """Return mock prefixes mapping all images to the same cluster."""
    return {i: prefix for i in range(n)}


def _run_pipeline(input_dir, output_dir, recursive=True, **kwargs):
    """Run organize_photos with mocked GPU pipeline."""
    # Count images to determine mock sizes
    from photo_organizer.scanner import ScanError, scan_images

    try:
        images = scan_images(str(input_dir), recursive=recursive)
    except ScanError:
        images = []
    n = len(images)

    mock_embeddings = np.random.rand(n, 128).astype(np.float32)
    mock_prefixes = _mock_prefixes(n)

    with (
        patch("photo_organizer.main.load_model", return_value=("m", "p", "cpu", "cpu")),
        patch("photo_organizer.main.extract_embeddings", return_value=(mock_embeddings, images)),
        patch("photo_organizer.main.recursive_cluster", return_value=mock_prefixes),
        patch("photo_organizer.main.classify_noise_with_clip"),
        patch("photo_organizer.main.generate_flat_gallery") as mock_gallery,
    ):
        organize_photos(
            str(input_dir),
            str(output_dir),
            recursive=recursive,
            cluster_algo="hdbscan",
            **kwargs,
        )
        return mock_gallery


class TestIntegrationBaseRun:
    """Default run — files moved, flat prefixed names, index.html exists."""

    def test_base_default_run(self, tmp_path):
        """Default flags: flat prefixed files + gallery generated."""
        _setup_images(tmp_path, ["a.jpg", "b.jpg"])
        output_dir = tmp_path / "out"

        mock_gallery = _run_pipeline(tmp_path / "src", output_dir)

        # Files moved with flat prefix
        assert (output_dir / "c0_a.jpg").exists()
        assert (output_dir / "c0_b.jpg").exists()
        # Gallery generated
        mock_gallery.assert_called_once()


class TestIntegrationNoGallery:
    """--no-gallery → files moved, no index.html."""

    def test_no_gallery_skips_gallery(self, tmp_path):
        """no_gallery=True: files moved, gallery NOT called."""
        _setup_images(tmp_path, ["photo.jpg"])
        output_dir = tmp_path / "out"

        mock_gallery = _run_pipeline(tmp_path / "src", output_dir, no_gallery=True)

        assert (output_dir / "c0_photo.jpg").exists()
        mock_gallery.assert_not_called()


class TestIntegrationNoRecursiveSearch:
    """--no-recursive-search → root images move, subdir images stay."""

    def test_non_recursive_root_only(self, tmp_path):
        """recursive=False: only root-level images processed."""
        _setup_images(tmp_path, ["root.jpg"])
        _setup_images(tmp_path, ["nested.jpg"], subdir="subdir")
        output_dir = tmp_path / "out"

        # Mock with only 1 image (root only)
        mock_embeddings = np.random.rand(1, 128).astype(np.float32)
        mock_prefixes = {0: "c0_"}

        with (
            patch("photo_organizer.main.load_model", return_value=("m", "p", "cpu", "cpu")),
            patch(
                "photo_organizer.main.extract_embeddings",
                return_value=(mock_embeddings, [tmp_path / "src" / "root.jpg"]),
            ),
            patch("photo_organizer.main.recursive_cluster", return_value=mock_prefixes),
            patch("photo_organizer.main.classify_noise_with_clip"),
            patch("photo_organizer.main.generate_flat_gallery"),
        ):
            organize_photos(
                str(tmp_path / "src"),
                str(output_dir),
                recursive=False,
                cluster_algo="hdbscan",
            )

        # Root image exported
        assert (output_dir / "c0_root.jpg").exists()
        # Subdir image NOT processed (still in source)
        assert (tmp_path / "src" / "subdir" / "nested.jpg").exists()


class TestIntegrationOutputModeFolders:
    """--output-mode folders → physical dirs created, no prefixes in names."""

    def test_folders_mode_creates_dirs(self, tmp_path):
        """output_mode='folders': subdirs created, original filenames."""
        _setup_images(tmp_path, ["photo.jpg"])
        output_dir = tmp_path / "out"

        _run_pipeline(tmp_path / "src", output_dir, output_mode="folders")

        # Physical dir created
        assert (output_dir / "c0" / "photo.jpg").exists()
        # No flat-prefixed file
        assert not (output_dir / "c0_photo.jpg").exists()


class TestIntegrationCollisionProtection:
    """Two duplicado.jpg files → both survive flat output."""

    def test_collision_both_survive(self, tmp_path):
        """Two files named duplicado.jpg with same prefix → both present."""
        _setup_images(tmp_path, ["duplicado.jpg"], subdir="dir1")
        _setup_images(tmp_path, ["duplicado.jpg"], subdir="dir2")
        output_dir = tmp_path / "out"

        # Mock with 2 images
        src_dir = tmp_path / "src"
        images = [src_dir / "dir1" / "duplicado.jpg", src_dir / "dir2" / "duplicado.jpg"]
        mock_embeddings = np.random.rand(2, 128).astype(np.float32)
        mock_prefixes = {0: "c0_", 1: "c0_"}

        with (
            patch("photo_organizer.main.load_model", return_value=("m", "p", "cpu", "cpu")),
            patch(
                "photo_organizer.main.extract_embeddings", return_value=(mock_embeddings, images)
            ),
            patch("photo_organizer.main.recursive_cluster", return_value=mock_prefixes),
            patch("photo_organizer.main.classify_noise_with_clip"),
            patch("photo_organizer.main.generate_flat_gallery"),
        ):
            organize_photos(
                str(src_dir),
                str(output_dir),
                cluster_algo="hdbscan",
            )

        # Both files present
        assert (output_dir / "c0_duplicado.jpg").exists()
        assert (output_dir / "c0_duplicado_1.jpg").exists()
        # Originals moved
        assert not images[0].exists()
        assert not images[1].exists()


class TestIntegrationKMeansMode:
    """--cluster-algo kmeans — flat single-pass clustering."""

    def test_kmeans_creates_clusters(self, tmp_path):
        """KMeans mode: all images assigned to flat c{k}_ prefixes."""
        from photo_organizer.main import organize_photos

        src_dir = tmp_path / "src"
        paths = []
        for i in range(6):
            p = src_dir / f"img_{i}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)
            paths.append(p)

        output_dir = tmp_path / "out"

        # Mock: 6 images → KMeans with k=2
        mock_embeddings = np.random.rand(6, 128).astype(np.float32)

        with (
            patch("photo_organizer.main.load_model", return_value=("m", "p", "cpu", "cpu")),
            patch("photo_organizer.main.extract_embeddings", return_value=(mock_embeddings, paths)),
            patch("photo_organizer.main.recursive_cluster") as mock_rc,
            patch("photo_organizer.main.classify_noise_with_clip"),
            patch("photo_organizer.main.generate_flat_gallery"),
        ):
            organize_photos(
                str(src_dir),
                str(output_dir),
                cluster_algo="kmeans",
                kmeans_k=2,
            )

        # Verify recursive_cluster was called with KMeans params
        mock_rc.assert_called_once()
        call_kwargs = mock_rc.call_args[1]
        assert call_kwargs.get("cluster_algo") == "kmeans"
        assert call_kwargs.get("kmeans_k") == 2

    def test_kmeans_skips_clip_fallback(self, tmp_path):
        """KMeans mode: clip_fallback is skipped since there's no noise concept."""
        from photo_organizer.main import organize_photos

        src_dir = tmp_path / "src"
        p = src_dir / "photo.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)

        output_dir = tmp_path / "out"

        mock_embeddings = np.random.rand(1, 128).astype(np.float32)
        images = [p]

        with (
            patch("photo_organizer.main.load_model", return_value=("m", "p", "cpu", "cpu")),
            patch(
                "photo_organizer.main.extract_embeddings", return_value=(mock_embeddings, images)
            ),
            patch("photo_organizer.main.recursive_cluster", return_value={0: "c0_"}),
            patch("photo_organizer.main.classify_noise_with_clip") as mock_clip,
            patch("photo_organizer.main.generate_flat_gallery"),
        ):
            organize_photos(
                str(src_dir),
                str(output_dir),
                cluster_algo="kmeans",
            )

        # CLIP fallback should be skipped for KMeans
        mock_clip.assert_not_called()
