"""Tests for models.py GPU validation error messages."""

from unittest.mock import patch

import pytest


class TestLoadModelGpuError:
    """Verify load_model() produces actionable error when CUDA is unavailable."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_cuda_unavailable_exits_with_code_1(self, _mock_cuda):
        """When CUDA is unavailable, load_model must exit with code 1."""
        from photo_organizer.models import load_model

        with pytest.raises(SystemExit) as exc_info:
            load_model("dinov2-large")
        assert exc_info.value.code == 1

    @patch("torch.cuda.is_available", return_value=False)
    def test_cuda_unavailable_mentions_rebuild_command(self, _mock_cuda, caplog):
        """Error message must include the rebuild command with cu124."""
        from photo_organizer.models import load_model

        with pytest.raises(SystemExit):
            load_model("dinov2-large")
        assert any("cu124" in record.message for record in caplog.records)

    @patch("torch.cuda.is_available", return_value=False)
    def test_cuda_unavailable_mentions_gpus_flag(self, _mock_cuda, caplog):
        """Error message must mention --gpus all."""
        from photo_organizer.models import load_model

        with pytest.raises(SystemExit):
            load_model("dinov2-large")
        assert any("--gpus all" in record.message for record in caplog.records)

    @patch("torch.cuda.is_available", return_value=False)
    def test_cuda_unavailable_mentions_nvidia_toolkit(self, _mock_cuda, caplog):
        """Error message must mention NVIDIA Container Toolkit."""
        from photo_organizer.models import load_model

        with pytest.raises(SystemExit):
            load_model("dinov2-large")
        assert any("NVIDIA" in record.message for record in caplog.records)

    @patch("torch.cuda.is_available", return_value=False)
    def test_cuda_unavailable_lists_possible_causes(self, _mock_cuda, caplog):
        """Error message must list numbered possible causes."""
        from photo_organizer.models import load_model

        with pytest.raises(SystemExit):
            load_model("dinov2-large")
        all_messages = " ".join(r.message for r in caplog.records)
        assert "Possible causes:" in all_messages

    @patch("torch.cuda.is_available", return_value=True)
    def test_cuda_available_no_error_message(self, _mock_cuda, caplog):
        """When CUDA IS available, no GPU error should appear in logs."""
        # Mock the model loading to avoid actually downloading models
        with patch("photo_organizer.models._load_dinov2") as mock_load:
            mock_load.return_value = (None, None, "cuda")
            from photo_organizer.models import load_model

            # Should not raise SystemExit
            load_model("dinov2-large")
        error_messages = [r.message for r in caplog.records if r.levelno >= 40]
        assert not any("CUDA is not available" in m for m in error_messages)
