"""Tests for flat_export collision protection — two files with same name in same cluster."""

from pathlib import Path

from photo_organizer.export import flat_export


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


class TestFlatExportCollisionProtection:
    """flat_export() handles collisions when two files have the same name."""

    def test_two_duplicado_files_both_survive(self, tmp_path):
        """Two duplicado.jpg files with same prefix → both in output, second gets _1."""
        src_dir = tmp_path / "src"
        src1 = src_dir / "dir1" / "duplicado.jpg"
        src2 = src_dir / "dir2" / "duplicado.jpg"
        _make_fake_jpeg(src1)
        _make_fake_jpeg(src2)

        output_dir = tmp_path / "out"
        prefixes = {0: "c0_", 1: "c0_"}

        count = flat_export([src1, src2], prefixes, output_dir)

        assert count == 2
        dest1 = output_dir / "c0_duplicado.jpg"
        dest2 = output_dir / "c0_duplicado_1.jpg"
        assert dest1.exists()
        assert dest2.exists()
        assert not src1.exists()
        assert not src2.exists()

    def test_sizes_match_originals(self, tmp_path):
        """Both output files have correct sizes matching originals."""
        src_dir = tmp_path / "src"
        src1 = src_dir / "d1" / "photo.jpg"
        src2 = src_dir / "d2" / "photo.jpg"
        src1.parent.mkdir(parents=True, exist_ok=True)
        src2.parent.mkdir(parents=True, exist_ok=True)
        src1.write_bytes(b"\xff\xd8\xff" + b"\x00" * 200)
        src2.write_bytes(b"\xff\xd8\xff" + b"\x00" * 300)

        output_dir = tmp_path / "out"
        prefixes = {0: "c0_", 1: "c0_"}

        flat_export([src1, src2], prefixes, output_dir)

        assert (output_dir / "c0_photo.jpg").stat().st_size == 203
        assert (output_dir / "c0_photo_1.jpg").stat().st_size == 303

    def test_three_same_names(self, tmp_path):
        """Three files with same name → _1, _2 suffixes."""
        src_dir = tmp_path / "src"
        paths = []
        for i in range(3):
            p = src_dir / f"sub{i}" / "same.jpg"
            _make_fake_jpeg(p)
            paths.append(p)

        output_dir = tmp_path / "out"
        prefixes = {0: "c0_", 1: "c0_", 2: "c0_"}

        flat_export(paths, prefixes, output_dir)

        assert (output_dir / "c0_same.jpg").exists()
        assert (output_dir / "c0_same_1.jpg").exists()
        assert (output_dir / "c0_same_2.jpg").exists()

    def test_different_names_no_collision(self, tmp_path):
        """Two files with different names → no suffix appended."""
        src_dir = tmp_path / "src"
        src1 = src_dir / "a.jpg"
        src2 = src_dir / "b.jpg"
        _make_fake_jpeg(src1)
        _make_fake_jpeg(src2)

        output_dir = tmp_path / "out"
        prefixes = {0: "c0_", 1: "c0_"}

        flat_export([src1, src2], prefixes, output_dir)

        assert (output_dir / "c0_a.jpg").exists()
        assert (output_dir / "c0_b.jpg").exists()
        # No _1 suffix on b.jpg
        assert not (output_dir / "c0_b_1.jpg").exists()
