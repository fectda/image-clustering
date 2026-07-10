"""Unit tests for cluster.py — UMAP + HDBSCAN + DBCV."""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def set_seeds():
    np.random.seed(42)
    torch.manual_seed(42)


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
            algo="hdbscan",
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
            algo="hdbscan",
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

        _, metrics = reduce_and_cluster(np.random.rand(50, 5120).astype(np.float64), algo="hdbscan")

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
            algo="hdbscan",
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


class TestCLIClipFallbackArgs:
    """CLI args: new defaults and --clip-fallback flag."""

    def test_default_min_cluster_size(self):
        from photo_organizer.cli import parse_args

        args = parse_args(["-i", "/tmp", "-o", "/tmp/out"])
        assert args.min_cluster_size == 3

    def test_default_min_samples(self):
        from photo_organizer.cli import parse_args

        args = parse_args(["-i", "/tmp", "-o", "/tmp/out"])
        assert args.min_samples == 1

    def test_default_clip_fallback_enabled(self):
        from photo_organizer.cli import parse_args

        args = parse_args(["-i", "/tmp", "-o", "/tmp/out"])
        assert args.clip_fallback is True

    def test_no_clip_fallback_flag(self):
        from photo_organizer.cli import parse_args

        args = parse_args(["-i", "/tmp", "-o", "/tmp/out", "--no-clip-fallback"])
        assert args.clip_fallback is False

    def test_custom_min_cluster_size(self):
        from photo_organizer.cli import parse_args

        args = parse_args(["-i", "/tmp", "-o", "/tmp/out", "--min-cluster-size", "10"])
        assert args.min_cluster_size == 10

    def test_custom_min_samples(self):
        from photo_organizer.cli import parse_args

        args = parse_args(["-i", "/tmp", "-o", "/tmp/out", "--min-samples", "5"])
        assert args.min_samples == 5


@pytest.fixture()
def mock_clip_deps():
    """Inject mock transformers CLIPModel/CLIPProcessor via sys.modules.

    Does NOT mock torch — CLIP classification needs real tensor operations.
    """
    mock_clip_model = MagicMock()
    mock_clip_processor = MagicMock()
    mock_transformers = MagicMock()
    mock_transformers.CLIPModel = mock_clip_model
    mock_transformers.CLIPProcessor = mock_clip_processor

    saved = {}
    for key in ("transformers",):
        saved[key] = sys.modules.get(key)

    sys.modules["transformers"] = mock_transformers

    yield mock_clip_model, mock_clip_processor, mock_transformers

    for key, original in saved.items():
        if original is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = original


class TestClassifyNoiseWithClip:
    """classify_noise_with_clip() — CLIP zero-shot classification of noise images."""

    def _create_test_images(self, tmp_path, n):
        """Create n small test JPEG images in tmp_path."""
        from PIL import Image

        paths = []
        for i in range(n):
            img = Image.new("RGB", (64, 64), color=(i * 30, i * 20, i * 10))
            p = tmp_path / f"img_{i}.jpg"
            img.save(p)
            paths.append(p)
        return paths

    def _setup_clip_mocks(self, mock_clip_deps, n_images):
        """Set up CLIP mocks with proper tensor returns for classification."""
        import torch

        mock_clip_model_cls, mock_clip_processor_cls, _ = mock_clip_deps

        # Mock CLIPModel
        mock_model = MagicMock()
        mock_clip_model_cls.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Mock get_image_features → return tensor based on pixel_values batch size
        def mock_get_image_features(**kwargs):
            pv = kwargs.get("pixel_values")
            batch = pv.shape[0] if pv is not None and hasattr(pv, "shape") else n_images
            return torch.randn(batch, 512)

        mock_model.get_image_features.side_effect = mock_get_image_features

        # Mock get_text_features → real tensor (7, 512) for 7 CLIP categories
        text_embeds = torch.randn(7, 512)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        mock_model.get_text_features.return_value = text_embeds

        # Mock CLIPProcessor
        mock_processor = MagicMock()
        mock_clip_processor_cls.from_pretrained.return_value = mock_processor

        # Processor returns dict-like object that supports **unpacking and .to()
        # Differentiate image calls (has images kwarg) vs text calls (no images kwarg)
        def mock_processor_call(*args, images=None, **kwargs):
            class FakeEncoding(dict):
                def to(self, dev):
                    return self

            if images is not None:
                batch = len(images)
                return FakeEncoding(
                    pixel_values=torch.randn(batch, 3, 224, 224),
                )
            else:
                return FakeEncoding(
                    input_ids=torch.randint(0, 1000, (1, 77)),
                )

        mock_processor.side_effect = mock_processor_call

        return mock_model, mock_processor

    def test_normal_classification(self, mock_clip_deps, tmp_path):
        """10 noise images → each gets c_misc_[category]_ prefix."""
        from photo_organizer.cluster import CATEGORY_NAMES, classify_noise_with_clip

        image_paths = self._create_test_images(tmp_path, 10)
        prefixes = {i: "" for i in range(10)}

        self._setup_clip_mocks(mock_clip_deps, 10)

        classify_noise_with_clip(image_paths, prefixes, device="cpu")

        # Each image should get a c_misc_[category]_ prefix
        for i in range(10):
            assert prefixes[i].startswith("c_misc_"), f"Image {i} prefix: {prefixes[i]}"
            category_part = prefixes[i].replace("c_misc_", "").rstrip("_")
            assert category_part in CATEGORY_NAMES, f"Unknown category: {category_part}"

    def test_zero_noise_images(self, mock_clip_deps):
        """Zero noise images → prefixes unchanged, no CLIP loaded."""
        from pathlib import Path

        from photo_organizer.cluster import classify_noise_with_clip

        mock_clip_model_cls, _, _ = mock_clip_deps

        # All images already have prefixes (no noise)
        prefixes = {0: "c0_", 1: "c0_c1_", 2: "c1_"}
        image_paths = [Path(f"/tmp/img_{i}.jpg") for i in range(3)]

        classify_noise_with_clip(image_paths, prefixes, device="cpu")

        # Prefixes unchanged
        assert prefixes == {0: "c0_", 1: "c0_c1_", 2: "c1_"}
        # CLIP model should NOT have been loaded
        mock_clip_model_cls.from_pretrained.assert_not_called()

    def test_model_load_failure(self, mock_clip_deps, caplog):
        """CLIP model load failure (Exception) → graceful skip, prefixes unchanged, warning logged."""
        from pathlib import Path

        from photo_organizer.cluster import classify_noise_with_clip

        mock_clip_model_cls, _, _ = mock_clip_deps

        prefixes = {0: "", 1: ""}
        image_paths = [Path("/tmp/img_0.jpg"), Path("/tmp/img_1.jpg")]

        # Make from_pretrained raise Exception
        mock_clip_model_cls.from_pretrained.side_effect = OSError("Model not found")

        classify_noise_with_clip(image_paths, prefixes, device="cpu")

        # Prefixes unchanged
        assert prefixes == {0: "", 1: ""}
        assert "Failed to load CLIP model" in caplog.text

    def test_single_noise_image(self, mock_clip_deps, tmp_path):
        """Single noise image → classified correctly."""
        from photo_organizer.cluster import CATEGORY_NAMES, classify_noise_with_clip

        image_paths = self._create_test_images(tmp_path, 1)
        prefixes = {0: ""}

        self._setup_clip_mocks(mock_clip_deps, 1)

        classify_noise_with_clip(image_paths, prefixes, device="cpu")

        assert prefixes[0].startswith("c_misc_")
        category_part = prefixes[0].replace("c_misc_", "").rstrip("_")
        assert category_part in CATEGORY_NAMES

    def test_mixed_prefixes(self, mock_clip_deps, tmp_path):
        """Some images noise, some clustered → only noise images get CLIP prefixes."""
        from photo_organizer.cluster import classify_noise_with_clip

        all_paths = self._create_test_images(tmp_path, 4)
        prefixes = {0: "c0_", 1: "", 2: "c1_", 3: ""}

        self._setup_clip_mocks(mock_clip_deps, 4)

        classify_noise_with_clip(all_paths, prefixes, device="cpu")

        # Clustered images unchanged
        assert prefixes[0] == "c0_"
        assert prefixes[2] == "c1_"
        # Noise images got CLIP prefix
        assert prefixes[1].startswith("c_misc_")
        assert prefixes[3].startswith("c_misc_")

    def test_corrupt_image_skipped(self, mock_clip_deps, tmp_path, caplog):
        """Corrupt image (Image.open fails) → skipped, valid images still classified, warning logged."""
        from photo_organizer.cluster import classify_noise_with_clip

        # One valid image
        from PIL import Image

        valid_path = tmp_path / "valid.jpg"
        Image.new("RGB", (64, 64)).save(valid_path)

        # One corrupt image (no file written)
        fake_path = tmp_path / "corrupt.jpg"

        image_paths = [valid_path, fake_path]
        prefixes = {0: "", 1: ""}

        # Setup mocks for 1 valid image (the corrupt one is skipped)
        self._setup_clip_mocks(mock_clip_deps, 1)

        classify_noise_with_clip(image_paths, prefixes, device="cpu")

        # Valid image got classified
        assert prefixes[0].startswith("c_misc_")
        # Corrupt image remained unchanged
        assert prefixes[1] == ""
        # Warning was logged
        assert "Skipping corrupt image" in caplog.text


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
        result = recursive_cluster(
            embeddings, paths, max_iterations=1, min_cluster_size=2, cluster_algo="hdbscan"
        )

        # All 6 images should get prefix "c0_"
        assert len(result) == 6
        for idx in range(6):
            assert result[idx] == "c0_"

        # Verify min_samples is threaded through to HDBSCAN
        call_kwargs = mock_hdbscan_mod.HDBSCAN.call_args[1]
        assert call_kwargs["min_samples"] == 1

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
        result = recursive_cluster(
            embeddings, paths, max_iterations=2, min_cluster_size=2, cluster_algo="hdbscan"
        )

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
        result = recursive_cluster(
            embeddings, paths, max_iterations=3, min_cluster_size=5, cluster_algo="hdbscan"
        )

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
        result = recursive_cluster(
            embeddings, paths, max_iterations=1, min_cluster_size=2, cluster_algo="hdbscan"
        )

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

        result = recursive_cluster(
            embeddings, paths, max_iterations=3, min_cluster_size=2, cluster_algo="hdbscan"
        )

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

        result = recursive_cluster(
            embeddings, paths, max_iterations=3, min_cluster_size=2, cluster_algo="hdbscan"
        )

        # Only 2 UMAP calls: depth 0 + depth 1 (no more after all-noise sub-group)
        assert call_count[0] == 2
        # Depth 1 subgroup all-noise → each inherits parent prefix 'c0_' from label_path [0]
        for idx in range(6):
            assert result[idx] == "c0_"
        # Noise at depth 0 → empty prefix
        for idx in range(6, 10):
            assert result[idx] == ""

    def test_empty_embeddings(self, mock_cluster_deps):
        """Empty array → empty dict."""
        from photo_organizer.cluster import recursive_cluster

        embeddings = np.array([]).reshape(0, 5120)
        paths = []

        result = recursive_cluster(
            embeddings, paths, max_iterations=3, min_cluster_size=2, cluster_algo="hdbscan"
        )

        assert result == {}


# ─── KMeans tests ───────────────────────────────────────────────


@pytest.fixture()
def mock_kmeans_deps():
    """Mock sklearn KMeans and umap for _cluster_kmeans testing."""
    mock_umap_mod = MagicMock()
    mock_kmeans_mod = MagicMock()

    saved = {}
    for key in ("umap", "sklearn.cluster"):
        saved[key] = sys.modules.get(key)

    sys.modules["umap"] = mock_umap_mod
    sys.modules["sklearn.cluster"] = mock_kmeans_mod

    yield mock_umap_mod, mock_kmeans_mod

    for key, original in saved.items():
        if original is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = original


class TestClusterKMeans:
    """_cluster_kmeans() with mocked sklearn KMeans — operates on already-reduced data."""

    def test_creates_k_clusters(self, mock_kmeans_deps):
        """100 samples, k=5 → 5 cluster labels."""
        from photo_organizer.cluster import _cluster_kmeans

        _, mock_kmeans_mod = mock_kmeans_deps

        # Mock KMeans
        mock_kmeans = MagicMock()
        mock_kmeans.fit_predict.return_value = np.array(
            [0] * 20 + [1] * 20 + [2] * 20 + [3] * 20 + [4] * 20
        )
        mock_kmeans_mod.KMeans.return_value = mock_kmeans

        # _cluster_kmeans takes already-reduced embeddings (UMAP done upstream)
        reduced = np.random.rand(100, 20).astype(np.float64)
        labels = _cluster_kmeans(reduced, k=5)

        assert len(labels) == 100
        assert len(set(labels)) == 5

    def test_small_dataset(self, mock_kmeans_deps):
        """5 samples, k=2 → 2 cluster labels."""
        from photo_organizer.cluster import _cluster_kmeans

        _, mock_kmeans_mod = mock_kmeans_deps

        mock_kmeans = MagicMock()
        mock_kmeans.fit_predict.return_value = np.array([0, 0, 1, 1, 0])
        mock_kmeans_mod.KMeans.return_value = mock_kmeans

        reduced = np.random.rand(5, 20).astype(np.float64)
        labels = _cluster_kmeans(reduced, k=2)

        assert len(labels) == 5
        assert len(set(labels)) == 2

    def test_k_greater_than_samples(self, mock_kmeans_deps):
        """k > n_samples → clamped to n_samples."""
        from photo_organizer.cluster import _cluster_kmeans

        _, mock_kmeans_mod = mock_kmeans_deps

        mock_kmeans = MagicMock()
        mock_kmeans.fit_predict.return_value = np.array([0, 1, 2])
        mock_kmeans_mod.KMeans.return_value = mock_kmeans

        reduced = np.random.rand(3, 20).astype(np.float64)
        labels = _cluster_kmeans(reduced, k=10)

        assert len(labels) == 3
        # KMeans clamped: min(10, max(2, 3-1)) = min(10, 2) = 2 (k must be < n_samples)
        call_kwargs = mock_kmeans_mod.KMeans.call_args[1]
        assert call_kwargs["n_clusters"] == 2


class TestRecursiveClusterKMeans:
    """recursive_cluster() with cluster_algo='kmeans' — recursive via shared loop."""

    def test_kmeans_single_pass(self, mock_cluster_deps):
        """KMeans mode with max_iterations=1: single clustering pass, flat c{k}_ prefixes."""
        from photo_organizer.cluster import recursive_cluster

        mock_umap_mod, _mock_hdbscan_mod, _ = mock_cluster_deps

        mock_reducer = MagicMock()
        mock_reducer.fit_transform.side_effect = lambda emb: np.random.rand(len(emb), 20).astype(
            np.float64
        )
        mock_umap_mod.UMAP.return_value = mock_reducer

        # KMeans shouldn't use HDBSCAN at all
        embeddings = np.random.rand(10, 5120).astype(np.float64)
        paths = [f"img_{i}.jpg" for i in range(10)]

        # Mock sklearn KMeans inside recursive_cluster
        import sys

        saved_kmeans = sys.modules.get("sklearn.cluster")
        mock_kmeans_mod = MagicMock()
        mock_kmeans = MagicMock()

        # Size-aware mock: return labels matching input size
        def mock_fit_predict(data):
            n = len(data)
            labels = np.zeros(n, dtype=int)
            if n > 3:
                labels[3:6] = 1
                labels[6:] = 2
            elif n > 1:
                labels[1:] = 1
            return labels

        mock_kmeans.fit_predict.side_effect = mock_fit_predict
        mock_kmeans_mod.KMeans.return_value = mock_kmeans
        sys.modules["sklearn.cluster"] = mock_kmeans_mod

        try:
            result = recursive_cluster(
                embeddings,
                paths,
                cluster_algo="kmeans",
                kmeans_k=3,
                max_iterations=1,
                min_cluster_size=2,
            )

            assert len(result) == 10
            # All images get flat cluster prefixes (no hierarchy, no noise)
            assert result[0] == "c0_"
            assert result[2] == "c0_"
            assert result[3] == "c1_"
            assert result[5] == "c1_"
            assert result[6] == "c2_"
            assert result[9] == "c2_"

            # HDBSCAN should NOT have been called
            _mock_hdbscan_mod.HDBSCAN.assert_not_called()
        finally:
            if saved_kmeans is not None:
                sys.modules["sklearn.cluster"] = saved_kmeans
            else:
                sys.modules.pop("sklearn.cluster", None)

    def test_kmeans_fewer_images(self, mock_cluster_deps):
        """KMeans mode with fewer images than k → clamped, max_iterations=1 for single pass."""
        from photo_organizer.cluster import recursive_cluster

        mock_umap_mod, _mock_hdbscan_mod, _ = mock_cluster_deps

        mock_reducer = MagicMock()
        mock_reducer.fit_transform.side_effect = lambda emb: np.random.rand(len(emb), 20).astype(
            np.float64
        )
        mock_umap_mod.UMAP.return_value = mock_reducer

        embeddings = np.random.rand(3, 5120).astype(np.float64)
        paths = [f"img_{i}.jpg" for i in range(3)]

        import sys

        saved_kmeans = sys.modules.get("sklearn.cluster")
        mock_kmeans_mod = MagicMock()
        mock_kmeans = MagicMock()

        # Size-aware mock for 3 items
        def mock_fit_predict(data):
            n = len(data)
            labels = np.zeros(n, dtype=int)
            if n > 1:
                labels[1:] = 1
            return labels

        mock_kmeans.fit_predict.side_effect = mock_fit_predict
        mock_kmeans_mod.KMeans.return_value = mock_kmeans
        sys.modules["sklearn.cluster"] = mock_kmeans_mod

        try:
            result = recursive_cluster(
                embeddings,
                paths,
                cluster_algo="kmeans",
                kmeans_k=8,
                max_iterations=1,
                min_cluster_size=2,
            )

            assert len(result) == 3
            # KMeans clamped to max(2, 3-1) = 2 clusters
            call_kwargs = mock_kmeans_mod.KMeans.call_args[1]
            assert call_kwargs["n_clusters"] == 2
            # Verify actual prefix values: mock returns [0, 1, 1]
            assert result[0] == "c0_"
            assert result[1] == "c1_"
            assert result[2] == "c1_"
        finally:
            if saved_kmeans is not None:
                sys.modules["sklearn.cluster"] = saved_kmeans
            else:
                sys.modules.pop("sklearn.cluster", None)

    def test_kmeans_recursive_nested_prefixes(self, mock_cluster_deps):
        """KMeans mode with max_iterations=2: nested prefixes c0_c1_."""
        from photo_organizer.cluster import recursive_cluster

        mock_umap_mod, _mock_hdbscan_mod, _ = mock_cluster_deps

        # Mock UMAP
        mock_reducer = MagicMock()
        mock_reducer.fit_transform.side_effect = lambda emb: np.random.rand(len(emb), 20).astype(
            np.float64
        )
        mock_umap_mod.UMAP.return_value = mock_reducer

        # Mock sklearn KMeans
        import sys

        saved_kmeans = sys.modules.get("sklearn.cluster")
        mock_kmeans_mod = MagicMock()
        mock_kmeans = MagicMock()
        # First call (depth 0): 20 images → cluster 0 (10), cluster 1 (10)
        # Second call (depth 1, sub-group of 10): cluster 0 (5), cluster 1 (5)
        call_count = [0]

        def mock_fit_predict(embeddings):
            call_count[0] += 1
            n = len(embeddings)
            if call_count[0] == 1:
                # depth 0: split into two equal clusters
                labels = np.zeros(n, dtype=int)
                labels[10:] = 1
                return labels
            else:
                # depth 1: split each sub-cluster further
                labels = np.zeros(n, dtype=int)
                labels[5:] = 1
                return labels

        mock_kmeans.fit_predict.side_effect = mock_fit_predict
        mock_kmeans_mod.KMeans.return_value = mock_kmeans
        sys.modules["sklearn.cluster"] = mock_kmeans_mod

        try:
            embeddings = np.random.rand(20, 5120).astype(np.float64)
            paths = [f"img_{i}.jpg" for i in range(20)]
            result = recursive_cluster(
                embeddings,
                paths,
                cluster_algo="kmeans",
                kmeans_k=4,
                max_iterations=2,
                min_cluster_size=2,
            )
            # Expect nested prefixes: first 10 images get c0_c0_ or c0_c1_, next 10 get c1_c0_ or c1_c1_
            # Since depth 1 splits each sub-cluster into 0 and 1, we expect:
            # indices 0-4: c0_c0_
            # indices 5-9: c0_c1_
            # indices 10-14: c1_c0_
            # indices 15-19: c1_c1_
            assert result[0] == "c0_c0_"
            assert result[4] == "c0_c0_"
            assert result[5] == "c0_c1_"
            assert result[9] == "c0_c1_"
            assert result[10] == "c1_c0_"
            assert result[14] == "c1_c0_"
            assert result[15] == "c1_c1_"
            assert result[19] == "c1_c1_"
            # All prefixes must be nested (at least 2 underscores: e.g. "c0_c1_")
            for idx in range(20):
                assert result[idx].count("_") >= 2
        finally:
            if saved_kmeans is not None:
                sys.modules["sklearn.cluster"] = saved_kmeans
            else:
                sys.modules.pop("sklearn.cluster", None)

    def test_kmeans_dynamic_k_clamping(self, mock_cluster_deps):
        """KMeans mode: K clamped when sub-group < kmeans_k.

        Uses kmeans_k=3 (non-default) to verify parameter forwarding
        from recursive_cluster → reduce_and_cluster → _cluster_kmeans.
        With 5 images: effective_k = min(3, max(2, 5-1)) = 3.
        """
        from photo_organizer.cluster import recursive_cluster

        mock_umap_mod, _mock_hdbscan_mod, _ = mock_cluster_deps

        mock_reducer = MagicMock()
        mock_reducer.fit_transform.side_effect = lambda emb: np.random.rand(len(emb), 20).astype(
            np.float64
        )
        mock_umap_mod.UMAP.return_value = mock_reducer

        import sys

        saved_kmeans = sys.modules.get("sklearn.cluster")
        mock_kmeans_mod = MagicMock()
        mock_kmeans = MagicMock()
        # 5 images, kmeans_k=3 → effective K = min(3, max(2, 5-1)) = 3
        mock_kmeans.fit_predict.return_value = np.array([0, 0, 1, 1, 2])
        mock_kmeans_mod.KMeans.return_value = mock_kmeans
        sys.modules["sklearn.cluster"] = mock_kmeans_mod

        try:
            embeddings = np.random.rand(5, 5120).astype(np.float64)
            paths = [f"img_{i}.jpg" for i in range(5)]
            result = recursive_cluster(
                embeddings,
                paths,
                cluster_algo="kmeans",
                kmeans_k=3,
                max_iterations=1,
                min_cluster_size=2,
            )
            # Verify KMeans was called with n_clusters=3 (forwarded, then clamped)
            call_kwargs = mock_kmeans_mod.KMeans.call_args[1]
            assert call_kwargs["n_clusters"] == 3
            # All images should have flat prefixes (max_iterations=1)
            assert result[0] == "c0_"
            assert result[4] == "c2_"
        finally:
            if saved_kmeans is not None:
                sys.modules["sklearn.cluster"] = saved_kmeans
            else:
                sys.modules.pop("sklearn.cluster", None)
