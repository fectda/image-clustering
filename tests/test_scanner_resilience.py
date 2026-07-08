"""Tests for scanner resilience on network paths."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestScannerResilience:
    """scan_images() skips unreadable files."""

    def test_skips_permission_error_file(self):
        """A file raising PermissionError during is_file() is skipped."""
        from photo_organizer.scanner import scan_images

        with (
            patch("pathlib.Path.is_dir", return_value=True),
            patch("pathlib.Path.rglob") as mock_rglob,
        ):
            mock_good = MagicMock(spec=Path)
            mock_good.is_file.return_value = True
            mock_good.suffix = ".jpg"
            mock_good.name = "good.jpg"

            mock_bad = MagicMock(spec=Path)
            mock_bad.is_file.side_effect = PermissionError("Permission denied")
            mock_bad.name = "bad.jpg"

            mock_rglob.return_value = [mock_good, mock_bad]

            result = scan_images(Path("/fake"))

            assert mock_good in result
            assert mock_bad not in result

    def test_skips_oserror_file(self):
        """A file raising OSError during is_file() is skipped."""
        from photo_organizer.scanner import scan_images

        with (
            patch("pathlib.Path.is_dir", return_value=True),
            patch("pathlib.Path.rglob") as mock_rglob,
        ):
            mock_good = MagicMock(spec=Path)
            mock_good.is_file.return_value = True
            mock_good.suffix = ".jpg"
            mock_good.name = "good.jpg"

            mock_bad = MagicMock(spec=Path)
            mock_bad.suffix = ".jpg"
            mock_bad.name = "broken.jpg"
            mock_bad.is_file.side_effect = OSError("Stale NFS handle")

            mock_rglob.return_value = [mock_good, mock_bad]

            result = scan_images(Path("/fake"))

            assert mock_good in result
            assert mock_bad not in result
