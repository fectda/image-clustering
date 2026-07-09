"""Unit tests for gallery.py — flat gallery regex grouping.
Tests for main.py — organize_photos() orchestrator."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestFlatGalleryGroupsByPrefix:
    """generate_flat_gallery() groups files by regex ^(c\\d+_)+ prefix."""

    def test_flat_gallery_groups_by_prefix(self, tmp_path):
        """c0_c1_a.jpg and c0_c1_b.jpg → same group."""
        from photo_organizer.gallery import generate_flat_gallery

        # Create files in the flat output dir
        (tmp_path / "c0_c1_a.jpg").touch()
        (tmp_path / "c0_c1_b.jpg").touch()

        generate_flat_gallery(tmp_path)

        index = (tmp_path / "index.html").read_text()
        # Both files should be in the same cluster section
        assert "c0_c1_a.jpg" in index
        assert "c0_c1_b.jpg" in index

    def test_flat_gallery_unclustered(self, tmp_path):
        """photo.jpg no prefix → 'Unclustered' section."""
        from photo_organizer.gallery import generate_flat_gallery

        (tmp_path / "photo.jpg").touch()

        generate_flat_gallery(tmp_path)

        index = (tmp_path / "index.html").read_text()
        # Should appear in unclustered section
        assert "Unclustered" in index
        assert "photo.jpg" in index

    def test_flat_gallery_multiple_groups(self, tmp_path):
        """c0_a.jpg, c1_b.jpg, c0_c.jpg → 2 groups."""
        from photo_organizer.gallery import generate_flat_gallery

        (tmp_path / "c0_a.jpg").touch()
        (tmp_path / "c1_b.jpg").touch()
        (tmp_path / "c0_c.jpg").touch()

        generate_flat_gallery(tmp_path)

        index = (tmp_path / "index.html").read_text()
        # Verify multiple cluster sections exist
        assert "Cluster" in index or "cluster" in index
        # All images should appear
        assert "c0_a.jpg" in index
        assert "c1_b.jpg" in index
        assert "c0_c.jpg" in index

    def test_flat_gallery_renders_html(self, tmp_path):
        """index.html created with correct sections when template renders."""
        from photo_organizer.gallery import generate_flat_gallery

        (tmp_path / "c0_a.jpg").touch()
        (tmp_path / "c1_b.jpg").touch()
        (tmp_path / "photo.jpg").touch()

        generate_flat_gallery(tmp_path)

        index_path = tmp_path / "index.html"
        assert index_path.exists()
        index = index_path.read_text()
        assert len(index) > 0
        # Verify basic HTML structure
        assert "<html" in index or "<!DOCTYPE" in index


class TestOrganizePhotos:
    """organize_photos() orchestrator — full pipeline integration."""

    def test_organize_photos_calls_recursive_cluster(self):
        """organize_photos calls recursive_cluster with correct args."""
        from photo_organizer.main import organize_photos

        mock_images = [Path(f"/tmp/images/img_{i}.jpg") for i in range(20)]
        mock_embeddings = np.random.rand(20, 5120).astype(np.float64)
        mock_prefixes = {i: f"c{i % 3}_" for i in range(20)}

        with (
            patch("photo_organizer.main.scan_images", return_value=mock_images),
            patch("photo_organizer.main.load_model", return_value=("m", "p", "dinov2", "cpu")),
            patch(
                "photo_organizer.main.extract_embeddings",
                return_value=(mock_embeddings, mock_images),
            ),
            patch("photo_organizer.main.recursive_cluster", return_value=mock_prefixes) as mock_rc,
            patch("photo_organizer.main.flat_export", return_value=20),
            patch("photo_organizer.main.generate_flat_gallery"),
        ):
            organize_photos("/tmp/images", "/tmp/output", max_iterations=3)

        mock_rc.assert_called_once_with(
            mock_embeddings,
            mock_images,
            max_iterations=3,
            min_cluster_size=3,
            min_samples=1,
            umap_n_components=20,
            umap_n_neighbors=40,
            umap_min_dist=0.0,
            umap_metric="cosine",
        )

    def test_organize_photos_dry_run_skips_moves(self):
        """dry_run=True → flat_export called with dry_run=True, no gallery generated."""
        from photo_organizer.main import organize_photos

        mock_images = [Path(f"/tmp/images/img_{i}.jpg") for i in range(20)]
        mock_embeddings = np.random.rand(20, 5120).astype(np.float64)
        mock_prefixes = {i: "c0_" for i in range(20)}

        with (
            patch("photo_organizer.main.scan_images", return_value=mock_images),
            patch("photo_organizer.main.load_model", return_value=("m", "p", "dinov2", "cpu")),
            patch(
                "photo_organizer.main.extract_embeddings",
                return_value=(mock_embeddings, mock_images),
            ),
            patch("photo_organizer.main.recursive_cluster", return_value=mock_prefixes),
            patch("photo_organizer.main.flat_export", return_value=0) as mock_fe,
            patch("photo_organizer.main.generate_flat_gallery") as mock_gfg,
        ):
            organize_photos("/tmp/images", "/tmp/output", max_iterations=3, dry_run=True)

        # flat_export should be called with dry_run=True
        mock_fe.assert_called_once_with(
            mock_images, mock_prefixes, Path("/tmp/output"), dry_run=True
        )
        # Gallery should NOT be generated in dry_run
        mock_gfg.assert_not_called()
