"""Unit tests for cluster.py — UMAP + HDBSCAN + DBCV."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestMinNeighbors:
    """min_neighbors() ensures n_neighbors < n_samples."""

    def test_normal_case(self):
        from photo_organizer.cluster import min_neighbors

        result = min_neighbors(40, 500)
        assert result == 40

    def test_clamps_to_samples_minus_one(self):
        from photo_organizer.cluster import min_neighbors

        result = min_neighbors(100, 10)
        assert result == 9

    def test_three_images(self):
        from photo_organizer.cluster import min_neighbors

        result = min_neighbors(40, 3)
        assert result == 2

    def test_two_images(self):
        from photo_organizer.cluster import min_neighbors

        result = min_neighbors(40, 2)
        assert result == 2  # minimum floor is 2

    def test_one_image(self):
        from photo_organizer.cluster import min_neighbors

        result = min_neighbors(40, 1)
        assert result == 2  # clamped: min(40, max(0, 2)) = max(2, 0) = 2

    def test_zero_samples(self):
        from photo_organizer.cluster import min_neighbors

        result = min_neighbors(40, 0)
        assert result == 2  # edge case: max(2, min(40, max(-1, 2))) = max(2, 2) = 2


@pytest.fixture()
def mock_cluster_deps():
    """Inject mock umap and hdbscan into sys.modules for lazy-import testing.

    cluster.py does ``import umap`` / ``import hdbscan`` inside the function
    body, so we pre-seed sys.modules with mocks before the function runs.
    """
    mock_umap_mod = MagicMock()
    mock_hdbscan_mod = MagicMock()
    mock_validity_mod = MagicMock()

    saved = {}
    for key in ("umap", "hdbscan", "hdbscan.validity"):
        saved[key] = sys.modules.get(key)

    sys.modules["umap"] = mock_umap_mod
    sys.modules["hdbscan"] = mock_hdbscan_mod
    sys.modules["hdbscan.validity"] = mock_validity_mod

    yield mock_umap_mod, mock_hdbscan_mod, mock_validity_mod

    # Restore original sys.modules state
    for key, original in saved.items():
        if original is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = original


class TestReduceAndCluster:
    """reduce_and_cluster() with mocked UMAP and HDBSCAN."""

    def test_normal_case(self, mock_cluster_deps):
        """Normal case: 100 samples, 5120d → 5 clusters, some outliers."""
        from photo_organizer.cluster import reduce_and_cluster

        mock_umap_mod, mock_hdbscan_mod, mock_validity_mod = mock_cluster_deps

        # Mock UMAP
        mock_reducer = MagicMock()
        reduced = np.random.rand(100, 20).astype(np.float64)
        mock_reducer.fit_transform.return_value = reduced
        mock_umap_mod.UMAP.return_value = mock_reducer

        # Mock HDBSCAN
        mock_clusterer = MagicMock()
        labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, -1] + [0] * 90)
        mock_clusterer.fit_predict.return_value = labels
        mock_hdbscan_mod.HDBSCAN.return_value = mock_clusterer

        # Mock DBCV validity_index
        mock_validity_mod.validity_index.return_value = 0.5

        labels_out, metrics = reduce_and_cluster(
            np.random.rand(100, 5120).astype(np.float64),
            n_components=20,
            n_neighbors=40,
        )

        assert len(labels_out) == 100
        assert metrics["n_clusters"] == 5
        assert metrics["n_outliers"] == 1
        assert isinstance(metrics["dbcv"], float)

    def test_all_noise(self, mock_cluster_deps):
        """All-noise edge case → DBCV=0.0 + warning."""
        from photo_organizer.cluster import reduce_and_cluster

        mock_umap_mod, mock_hdbscan_mod, _ = mock_cluster_deps

        mock_reducer = MagicMock()
        reduced = np.random.rand(50, 20).astype(np.float64)
        mock_reducer.fit_transform.return_value = reduced
        mock_umap_mod.UMAP.return_value = mock_reducer

        mock_clusterer = MagicMock()
        mock_clusterer.fit_predict.return_value = np.full(50, -1)
        mock_hdbscan_mod.HDBSCAN.return_value = mock_clusterer

        labels_out, metrics = reduce_and_cluster(
            np.random.rand(50, 5120).astype(np.float64),
            min_cluster_size=15,
        )

        assert np.all(labels_out == -1)
        assert metrics["n_clusters"] == 0
        assert metrics["n_outliers"] == 50
        assert metrics["dbcv"] == 0.0

    def test_single_cluster(self, mock_cluster_deps):
        """Single cluster → DBCV=0.0 (undefined for <2 clusters)."""
        from photo_organizer.cluster import reduce_and_cluster

        mock_umap_mod, mock_hdbscan_mod, _ = mock_cluster_deps

        mock_reducer = MagicMock()
        mock_reducer.fit_transform.return_value = np.random.rand(50, 20).astype(np.float64)
        mock_umap_mod.UMAP.return_value = mock_reducer

        mock_clusterer = MagicMock()
        mock_clusterer.fit_predict.return_value = np.zeros(50, dtype=int)
        mock_hdbscan_mod.HDBSCAN.return_value = mock_clusterer

        _, metrics = reduce_and_cluster(np.random.rand(50, 5120).astype(np.float64))

        assert metrics["n_clusters"] == 1
        assert metrics["dbcv"] == 0.0

    def test_small_dataset(self, mock_cluster_deps):
        """Small dataset (5 samples) works with min_neighbors guard."""
        from photo_organizer.cluster import reduce_and_cluster

        mock_umap_mod, mock_hdbscan_mod, _ = mock_cluster_deps

        mock_reducer = MagicMock()
        mock_reducer.fit_transform.return_value = np.random.rand(5, 20).astype(np.float64)
        mock_umap_mod.UMAP.return_value = mock_reducer

        mock_clusterer = MagicMock()
        mock_clusterer.fit_predict.return_value = np.array([0, 0, 1, 1, -1])
        mock_hdbscan_mod.HDBSCAN.return_value = mock_clusterer

        labels_out, metrics = reduce_and_cluster(
            np.random.rand(5, 5120).astype(np.float64),
            n_neighbors=40,  # would be 40 but min_neighbors caps to 4
        )

        # Verify UMAP was called with adjusted n_neighbors
        call_kwargs = mock_umap_mod.UMAP.call_args[1]
        assert call_kwargs["n_neighbors"] == 4  # min(40, 5-1) = 4
        assert len(labels_out) == 5
        assert metrics["n_clusters"] == 2
