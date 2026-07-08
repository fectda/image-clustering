"""Unit tests for embeddings.py — extract_hybrid_embeddings."""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest


class TestExtractHybridEmbeddings:
    """extract_hybrid_embeddings() with mocked _extract_single_backend."""

    N = 10  # small image count for tests

    @patch("photo_organizer.embeddings._extract_single_backend")
    def test_happy_path(self, mock_extract):
        """DINOv2 (N,1024) + Qwen3 (N,4096) → (N,5120) L2-normalized."""
        from photo_organizer.embeddings import extract_hybrid_embeddings

        dinov2_emb = np.random.rand(self.N, 1024).astype(np.float32)
        qwen3_emb = np.random.rand(self.N, 4096).astype(np.float32)
        paths = ["img"] * self.N
        mock_extract.side_effect = [(dinov2_emb, paths), (qwen3_emb, paths)]

        models = {"dinov2": (MagicMock(), MagicMock()), "qwen3": (MagicMock(), MagicMock())}
        result, result_paths = extract_hybrid_embeddings(
            models, ["img"] * self.N, "cpu", batch_size=32
        )

        assert result.shape == (self.N, 5120)
        assert len(result_paths) == self.N
        # Verify L2-normalized: each row should have norm ≈ 1.0
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    @patch("photo_organizer.embeddings._extract_single_backend")
    def test_dinov2_empty_exits(self, mock_extract):
        """DINOv2 returns empty array → sys.exit(1)."""
        from photo_organizer.embeddings import extract_hybrid_embeddings

        mock_extract.side_effect = [
            (np.array([]), []),
            (np.random.rand(self.N, 4096), ["img"] * self.N),
        ]

        models = {"dinov2": (MagicMock(), MagicMock()), "qwen3": (MagicMock(), MagicMock())}
        with pytest.raises(SystemExit) as exc_info:
            extract_hybrid_embeddings(models, ["img"] * self.N, "cpu")
        assert exc_info.value.code == 1

    @patch("photo_organizer.embeddings._extract_single_backend")
    def test_qwen3_empty_exits(self, mock_extract):
        """Qwen3 returns empty array → sys.exit(1)."""
        from photo_organizer.embeddings import extract_hybrid_embeddings

        mock_extract.side_effect = [
            (np.random.rand(self.N, 1024), ["img"] * self.N),
            (np.array([]), []),
        ]

        models = {"dinov2": (MagicMock(), MagicMock()), "qwen3": (MagicMock(), MagicMock())}
        with pytest.raises(SystemExit) as exc_info:
            extract_hybrid_embeddings(models, ["img"] * self.N, "cpu")
        assert exc_info.value.code == 1

    @patch("photo_organizer.embeddings._extract_single_backend")
    def test_image_count_mismatch_exits(self, mock_extract):
        """DINOv2 and Qwen3 return different N → sys.exit(1)."""
        from photo_organizer.embeddings import extract_hybrid_embeddings

        mock_extract.side_effect = [
            (np.random.rand(10, 1024), ["img"] * 10),
            (np.random.rand(9, 4096), ["img"] * 9),
        ]

        models = {"dinov2": (MagicMock(), MagicMock()), "qwen3": (MagicMock(), MagicMock())}
        with pytest.raises(SystemExit) as exc_info:
            extract_hybrid_embeddings(models, ["img"] * 10, "cpu")
        assert exc_info.value.code == 1
