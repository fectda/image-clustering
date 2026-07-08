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
        assert result == 1  # fewer than 2 samples → return 1

    def test_zero_samples(self):
        from photo_organizer.cluster import min_neighbors

        result = min_neighbors(40, 0)
        assert result == 1  # fewer than 2 samples → return 1


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


class TestBuildPrefix:
    """_build_prefix() generates hierarchical cluster prefix strings."""

    def test_single_level(self):
        from photo_organizer.cluster import _build_prefix

        assert _build_prefix(label_path=[5]) == "c5_"

    def test_multi_level(self):
        from photo_organizer.cluster import _build_prefix

        assert _build_prefix(label_path=[5, 3]) == "c5_c3_"

    def test_empty_label_path(self):
        from photo_organizer.cluster import _build_prefix

        assert _build_prefix(label_path=[]) == ""

    def test_three_levels(self):
        from photo_organizer.cluster import _build_prefix

        assert _build_prefix(label_path=[1, 2, 3]) == "c1_c2_c3_"


class TestCliMaxIterations:
    """--max-iterations arg added to CLI."""

    def test_default_value(self):
        from photo_organizer.cli import parse_args

        args = parse_args(["-i", "/tmp", "-o", "/tmp/out"])
        assert args.max_iterations == 3

    def test_custom_value(self):
        from photo_organizer.cli import parse_args

        args = parse_args(["-i", "/tmp", "-o", "/tmp/out", "--max-iterations", "5"])
        assert args.max_iterations == 5


class TestRecursiveCluster:
    """recursive_cluster() — stack-based iterative clustering."""

    def _mock_umap(self, mock_umap_mod):
        """Create a UMAP mock that returns correct shape for input size."""
        mock_reducer = MagicMock()
        mock_reducer.fit_transform.side_effect = lambda emb: np.random.rand(len(emb), 20).astype(
            np.float64
        )
        mock_umap_mod.UMAP.return_value = mock_reducer
        return mock_reducer

    def test_single_level(self, mock_cluster_deps):
        """All images in one cluster → prefixes 'c0_' for all."""
        from photo_organizer.cluster import recursive_cluster

        mock_umap_mod, mock_hdbscan_mod, _ = mock_cluster_deps
        mock_reducer = self._mock_umap(mock_umap_mod)

        mock_clusterer = MagicMock()
        # All 6 images in cluster 0
        mock_clusterer.fit_predict.return_value = np.array([0, 0, 0, 0, 0, 0])
        mock_hdbscan_mod.HDBSCAN.return_value = mock_clusterer

        embeddings = np.random.rand(6, 5120).astype(np.float64)
        paths = [f"img_{i}.jpg" for i in range(6)]

        # max_iterations=1: only depth 0 processed → terminal with c0_
        result = recursive_cluster(embeddings, paths, max_iterations=1, min_cluster_size=2)

        # All 6 images should get prefix "c0_"
        assert len(result) == 6
        for idx in range(6):
            assert result[idx] == "c0_"

    def test_two_levels(self, mock_cluster_deps):
        """Parent cluster 0 → child cluster 1 → prefixes 'c0_c1_' for children."""
        from photo_organizer.cluster import recursive_cluster

        mock_umap_mod, mock_hdbscan_mod, _ = mock_cluster_deps
        mock_reducer = self._mock_umap(mock_umap_mod)

        call_count = [0]

        def mock_fit_predict(embeddings):
            call_count[0] += 1
            n = len(embeddings)
            if call_count[0] == 1:
                # First call (depth 0): 10 images → cluster 0 (8), noise (2)
                labels = np.zeros(n, dtype=int)
                labels[-2:] = -1  # last 2 are noise
                return labels
            else:
                # Second call (depth 1, sub-group of 8): cluster 0 (4), cluster 1 (4)
                labels = np.zeros(n, dtype=int)
                labels[4:] = 1
                return labels

        mock_clusterer = MagicMock()
        mock_clusterer.fit_predict.side_effect = mock_fit_predict
        mock_hdbscan_mod.HDBSCAN.return_value = mock_clusterer

        embeddings = np.random.rand(10, 5120).astype(np.float64)
        paths = [f"img_{i}.jpg" for i in range(10)]

        # max_iterations=2: depth 0 + depth 1 processed, depth 2 is terminal
        result = recursive_cluster(embeddings, paths, max_iterations=2, min_cluster_size=2)

        # First 4 images (in child cluster 0 at depth 1): "c0_c0_"
        # Next 4 images (in child cluster 1 at depth 1): "c0_c1_"
        # Last 2 images (noise at depth 0): ""
        assert result[0] == "c0_c0_"
        assert result[3] == "c0_c0_"
        assert result[4] == "c0_c1_"
        assert result[7] == "c0_c1_"
        assert result[8] == ""
        assert result[9] == ""

    def test_early_stopping(self, mock_cluster_deps):
        """Sub-group < min_cluster_size → skipped, no UMAP call for that group."""
        from photo_organizer.cluster import recursive_cluster

        mock_umap_mod, mock_hdbscan_mod, _ = mock_cluster_deps
        mock_reducer = self._mock_umap(mock_umap_mod)

        mock_clusterer = MagicMock()
        # depth 0: 10 images → cluster 0 (3), cluster 1 (3), noise (4)
        mock_clusterer.fit_predict.return_value = np.array([0, 0, 0, 1, 1, 1, -1, -1, -1, -1])
        mock_hdbscan_mod.HDBSCAN.return_value = mock_clusterer

        embeddings = np.random.rand(10, 5120).astype(np.float64)
        paths = [f"img_{i}.jpg" for i in range(10)]

        # min_cluster_size=5: cluster 0 (3) and cluster 1 (3) are below threshold
        # → no children pushed → only depth 0 processed
        result = recursive_cluster(embeddings, paths, max_iterations=3, min_cluster_size=5)

        # Only 1 UMAP call (depth 0), clusters 0 and 1 both < min_cluster_size=5
        # so they are terminal with their depth-0 prefix
        assert mock_reducer.fit_transform.call_count == 1
        assert result[0] == "c0_"
        assert result[2] == "c0_"
        assert result[3] == "c1_"
        assert result[5] == "c1_"
        # Noise images
        assert result[6] == ""
        assert result[9] == ""

    def test_max_depth(self, mock_cluster_deps):
        """max_iterations=1 → only depth 0 processed, no recursive calls."""
        from photo_organizer.cluster import recursive_cluster

        mock_umap_mod, mock_hdbscan_mod, _ = mock_cluster_deps
        mock_reducer = self._mock_umap(mock_umap_mod)

        mock_clusterer = MagicMock()
        # depth 0: cluster 0 (5), cluster 1 (5)
        mock_clusterer.fit_predict.return_value = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        mock_hdbscan_mod.HDBSCAN.return_value = mock_clusterer

        embeddings = np.random.rand(10, 5120).astype(np.float64)
        paths = [f"img_{i}.jpg" for i in range(10)]

        # max_iterations=1: only depth 0 processed (depth 0 < max_iterations=1)
        result = recursive_cluster(embeddings, paths, max_iterations=1, min_cluster_size=2)

        # Only 1 UMAP call
        assert mock_reducer.fit_transform.call_count == 1
        assert result[0] == "c0_"
        assert result[4] == "c0_"
        assert result[5] == "c1_"
        assert result[9] == "c1_"

    def test_noise_prefix(self, mock_cluster_deps):
        """Label -1 at any depth → empty prefix."""
        from photo_organizer.cluster import recursive_cluster

        mock_umap_mod, mock_hdbscan_mod, _ = mock_cluster_deps
        mock_reducer = self._mock_umap(mock_umap_mod)

        mock_clusterer = MagicMock()
        # All noise at depth 0
        mock_clusterer.fit_predict.return_value = np.full(6, -1, dtype=int)
        mock_hdbscan_mod.HDBSCAN.return_value = mock_clusterer

        embeddings = np.random.rand(6, 5120).astype(np.float64)
        paths = [f"img_{i}.jpg" for i in range(6)]

        result = recursive_cluster(embeddings, paths, max_iterations=3, min_cluster_size=2)

        # All noise → all get empty prefix
        for idx in range(6):
            assert result[idx] == ""

    def test_all_noise_subgroup(self, mock_cluster_deps):
        """All -1 in sub-group → no children pushed, no infinite loop."""
        from photo_organizer.cluster import recursive_cluster

        mock_umap_mod, mock_hdbscan_mod, _ = mock_cluster_deps
        mock_reducer = self._mock_umap(mock_umap_mod)

        call_count = [0]

        def mock_fit_predict(embeddings):
            call_count[0] += 1
            n = len(embeddings)
            if call_count[0] == 1:
                # First call: 10 images → cluster 0 (6), noise (4)
                labels = np.full(n, -1, dtype=int)
                labels[:6] = 0
                return labels
            else:
                # Second call (depth 1, sub-group of 6): all noise
                return np.full(n, -1, dtype=int)

        mock_clusterer = MagicMock()
        mock_clusterer.fit_predict.side_effect = mock_fit_predict
        mock_hdbscan_mod.HDBSCAN.return_value = mock_clusterer

        embeddings = np.random.rand(10, 5120).astype(np.float64)
        paths = [f"img_{i}.jpg" for i in range(10)]

        result = recursive_cluster(embeddings, paths, max_iterations=3, min_cluster_size=2)

        # Only 2 UMAP calls: depth 0 + depth 1 (no more after all-noise sub-group)
        assert call_count[0] == 2
        # All images in cluster 0 at depth 1 are noise → empty prefix
        for idx in range(6):
            assert result[idx] == ""
        # Noise at depth 0 → empty prefix
        for idx in range(6, 10):
            assert result[idx] == ""

    def test_empty_embeddings(self, mock_cluster_deps):
        """Empty array → empty dict."""
        from photo_organizer.cluster import recursive_cluster

        embeddings = np.array([]).reshape(0, 5120)
        paths = []

        result = recursive_cluster(embeddings, paths, max_iterations=3, min_cluster_size=2)

        assert result == {}
