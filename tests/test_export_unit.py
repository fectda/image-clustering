"""Unit tests for export.py — export_clusters and flat_export."""

import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

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


class TestFlatExport:
    """flat_export() with mocked shutil.move."""

    def test_flat_export_basic(self, tmp_path):
        """Files moved with prefix prepended, shutil.move called with correct dest path."""
        from photo_organizer.export import flat_export

        image_paths = [Path("/fake/photo.jpg"), Path("/fake/landscape.jpg")]
        prefixes = {0: "c0_", 1: "c0_c1_"}

        with patch("photo_organizer.export.shutil.move") as mock_move:
            count = flat_export(image_paths, prefixes, tmp_path, dry_run=False)

        assert count == 2
        assert mock_move.call_count == 2
        # Verify correct dest paths
        first_call_src, first_call_dst = mock_move.call_args_list[0][0]
        assert first_call_src == str(image_paths[0])
        assert first_call_dst == str(tmp_path / "c0_photo.jpg")

        second_call_src, second_call_dst = mock_move.call_args_list[1][0]
        assert second_call_src == str(image_paths[1])
        assert second_call_dst == str(tmp_path / "c0_c1_landscape.jpg")

    def test_flat_export_collision(self, tmp_path):
        """Second same-name file gets _1 suffix appended to stem."""
        from photo_organizer.export import flat_export

        # Two files with the SAME name but different source paths
        image_paths = [Path("/fake/dir/photo.jpg"), Path("/fake/other/photo.jpg")]
        prefixes = {0: "c0_", 1: "c0_"}  # Same prefix, same filename → collision

        with patch("photo_organizer.export.shutil.move") as mock_move:
            count = flat_export(image_paths, prefixes, tmp_path, dry_run=False)

        assert count == 2
        # First file → c0_photo.jpg, second file → collision → c0_photo_1.jpg
        calls = mock_move.call_args_list
        first_dst = calls[0][0][1]
        second_dst = calls[1][0][1]
        assert first_dst == str(tmp_path / "c0_photo.jpg")
        assert second_dst == str(tmp_path / "c0_photo_1.jpg")

    def test_flat_export_dry_run(self, tmp_path):
        """shutil.move never called, returns 0."""
        from photo_organizer.export import flat_export

        image_paths = [Path("/fake/photo.jpg")]
        prefixes = {0: "c0_"}

        with patch("photo_organizer.export.shutil.move") as mock_move:
            count = flat_export(image_paths, prefixes, tmp_path, dry_run=True)

        mock_move.assert_not_called()
        assert count == 0

    def test_flat_export_cross_device(self, tmp_path):
        """shutil.move raises OSError → fallback to _verified_copy + os.remove."""
        from photo_organizer.export import flat_export

        image_paths = [Path("/fake/photo.jpg")]
        prefixes = {0: "c0_"}

        with (
            patch(
                "photo_organizer.export.shutil.move",
                side_effect=OSError("Invalid cross-device link"),
            ),
            patch("photo_organizer.export._verified_copy") as mock_verified,
            patch("photo_organizer.export.os.remove") as mock_remove,
        ):
            count = flat_export(image_paths, prefixes, tmp_path, dry_run=False)

        assert count == 1
        mock_verified.assert_called_once()
        src_copy, dst_copy = mock_verified.call_args[0]
        assert src_copy == image_paths[0]
        assert dst_copy == tmp_path / "c0_photo.jpg"
        mock_remove.assert_called_once_with(str(image_paths[0]))

    def test_flat_export_returns_count(self, tmp_path):
        """Correct integer count returned after moves."""
        from photo_organizer.export import flat_export

        image_paths = [Path(f"/fake/img_{i}.jpg") for i in range(5)]
        prefixes = {i: f"c{i % 2}_" for i in range(5)}

        with patch("photo_organizer.export.shutil.move"):
            count = flat_export(image_paths, prefixes, tmp_path, dry_run=False)

        assert count == 5

    def test_flat_export_empty_prefix(self, tmp_path):
        """Unclustered file (empty prefix) keeps original filename."""
        from photo_organizer.export import flat_export

        image_paths = [Path("/fake/photo.jpg")]
        prefixes = {0: ""}  # Empty prefix = unclustered

        with patch("photo_organizer.export.shutil.move") as mock_move:
            count = flat_export(image_paths, prefixes, tmp_path, dry_run=False)

        assert count == 1
        dest = mock_move.call_args[0][1]
        assert dest == str(tmp_path / "photo.jpg")


class TestVerifiedCopy:
    """_verified_copy() — copy with size verification and retry."""

    def test_size_mismatch_raises_oserror(self, tmp_path):
        """copy2 succeeds but dest size differs → OSError, source NOT removed."""
        from photo_organizer.export import _verified_copy

        src = tmp_path / "source.jpg"
        dest = tmp_path / "dest.jpg"
        src.write_bytes(b"\x00" * 100)

        def stat_side_effect(path_self):
            """Return different sizes for src vs dest based on the object."""
            s = MagicMock()
            # Use object identity to distinguish src from dest
            if path_self is src:
                s.st_size = 100
            else:
                s.st_size = 50
            return s

        with (
            patch("photo_organizer.export.shutil.copy2"),
            patch("photo_organizer.export.os.remove") as mock_remove,
            patch.object(Path, "stat", stat_side_effect),
        ):
            with pytest.raises(OSError, match="Size mismatch"):
                _verified_copy(src, dest)

        mock_remove.assert_not_called()

    def test_copy_oserror_preserves_source(self, tmp_path):
        """copy2 raises OSError → source file is NOT deleted."""
        from photo_organizer.export import _verified_copy

        src = tmp_path / "source.jpg"
        dest = tmp_path / "dest.jpg"
        src.write_bytes(b"\x00" * 100)

        with (
            patch(
                "photo_organizer.export.shutil.copy2",
                side_effect=OSError("Network unreachable"),
            ),
            patch("photo_organizer.export.os.remove") as mock_remove,
        ):
            with pytest.raises(OSError, match="Network unreachable"):
                _verified_copy(src, dest)

        mock_remove.assert_not_called()

    def test_retry_succeeds_on_second_attempt(self, tmp_path):
        """First copy2 fails, second succeeds with correct size → no exception."""
        from photo_organizer.export import _verified_copy

        src = tmp_path / "source.jpg"
        dest = tmp_path / "dest.jpg"
        src.write_bytes(b"\x00" * 100)

        call_count = 0

        def copy2_side_effect(src_str, dest_str):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("Transient network error")
            # Second call succeeds — write actual data so stat works
            Path(dest_str).write_bytes(b"\x00" * 100)

        with (
            patch("photo_organizer.export.shutil.copy2", side_effect=copy2_side_effect),
            patch("photo_organizer.export.time.sleep"),
            patch("photo_organizer.export.os.remove") as mock_remove,
        ):
            # Should NOT raise — second attempt succeeds
            _verified_copy(src, dest, retries=2)

        assert call_count == 2
        mock_remove.assert_not_called()
