"""Unit tests for export.py — export_clusters."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


class TestExportClusters:
    """export_clusters() with mocked shutil.copy2."""

    def test_export_with_unclustered(self, tmp_path):
        """export_unclustered=True: noise images (-1) are included in groups."""
        from photo_organizer.export import export_clusters

        image_paths = [Path(f"/fake/img_{i}.jpg") for i in range(5)]
        labels = np.array([0, 0, 1, -1, -1])

        with patch("shutil.copy2"):
            groups = export_clusters(
                image_paths, labels, tmp_path, dry_run=False, export_unclustered=True
            )

        assert -1 in groups, "Noise images should be in groups"
        assert len(groups[-1]) == 2
        total = sum(len(v) for v in groups.values())
        assert total == 5

    def test_export_without_unclustered(self, tmp_path):
        """export_unclustered=False: noise images (-1) are filtered out."""
        from photo_organizer.export import export_clusters

        image_paths = [Path(f"/fake/img_{i}.jpg") for i in range(5)]
        labels = np.array([0, 0, 1, -1, -1])

        with patch("shutil.copy2"):
            groups = export_clusters(
                image_paths, labels, tmp_path, dry_run=False, export_unclustered=False
            )

        assert -1 not in groups, "Noise images should be excluded"
        total = sum(len(v) for v in groups.values())
        assert total == 3

    def test_dry_run_does_not_copy(self, tmp_path):
        """dry_run=True: shutil.copy2 should not be called."""
        from photo_organizer.export import export_clusters

        image_paths = [Path(f"/fake/img_{i}.jpg") for i in range(5)]
        labels = np.array([0, 0, 1, -1, -1])

        with patch("shutil.copy2") as mock_copy:
            groups = export_clusters(
                image_paths, labels, tmp_path, dry_run=True, export_unclustered=True
            )

        mock_copy.assert_not_called()
        assert -1 in groups
