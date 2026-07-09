"""Tests for folders_export() in export.py."""

from pathlib import Path

from photo_organizer.export import folders_export


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


class TestFoldersExport:
    """folders_export() creates subdirectory hierarchy from prefixes."""

    def test_creates_nested_dirs(self, tmp_path):
        """Prefix 'c1_c2_' creates output/c1/c2/IMG.jpg hierarchy."""
        src_dir = tmp_path / "src"
        src = src_dir / "photo.jpg"
        _make_fake_jpeg(src)

        output_dir = tmp_path / "out"
        prefixes = {0: "c1_c2_"}

        count = folders_export([src], prefixes, output_dir)

        assert count == 1
        dest = output_dir / "c1" / "c2" / "photo.jpg"
        assert dest.exists()
        assert not src.exists()

    def test_original_filename_no_prefix(self, tmp_path):
        """Images placed with original filenames, no prefix in name."""
        src_dir = tmp_path / "src"
        src = src_dir / "landscape.jpg"
        _make_fake_jpeg(src)

        output_dir = tmp_path / "out"
        prefixes = {0: "c0_"}

        folders_export([src], prefixes, output_dir)

        dest = output_dir / "c0" / "landscape.jpg"
        assert dest.exists()
        assert dest.name == "landscape.jpg"

    def test_unclustered_goes_to_unclustered_dir(self, tmp_path):
        """Empty prefix places images in output/unclustered/."""
        src_dir = tmp_path / "src"
        src = src_dir / "noise.jpg"
        _make_fake_jpeg(src)

        output_dir = tmp_path / "out"
        prefixes = {0: ""}

        folders_export([src], prefixes, output_dir)

        dest = output_dir / "unclustered" / "noise.jpg"
        assert dest.exists()

    def test_single_level_prefix(self, tmp_path):
        """Prefix 'c0_' creates output/c0/IMG.jpg."""
        src_dir = tmp_path / "src"
        src = src_dir / "img.jpg"
        _make_fake_jpeg(src)

        output_dir = tmp_path / "out"
        prefixes = {0: "c0_"}

        folders_export([src], prefixes, output_dir)

        dest = output_dir / "c0" / "img.jpg"
        assert dest.exists()

    def test_multiple_files_different_prefixes(self, tmp_path):
        """Multiple files with different prefixes create separate hierarchies."""
        src_dir = tmp_path / "src"
        src1 = src_dir / "a.jpg"
        src2 = src_dir / "b.jpg"
        _make_fake_jpeg(src1)
        _make_fake_jpeg(src2)

        output_dir = tmp_path / "out"
        prefixes = {0: "c0_", 1: "c1_c2_"}

        count = folders_export([src1, src2], prefixes, output_dir)

        assert count == 2
        assert (output_dir / "c0" / "a.jpg").exists()
        assert (output_dir / "c1" / "c2" / "b.jpg").exists()

    def test_returns_count(self, tmp_path):
        """Returns integer count of moved files."""
        src_dir = tmp_path / "src"
        paths = []
        for i in range(5):
            p = src_dir / f"img_{i}.jpg"
            _make_fake_jpeg(p)
            paths.append(p)

        output_dir = tmp_path / "out"
        prefixes = {i: f"c{i % 2}_" for i in range(5)}

        count = folders_export(paths, prefixes, output_dir)

        assert count == 5

    def test_dry_run_does_not_move(self, tmp_path):
        """dry_run=True: no files moved, returns 0."""
        src_dir = tmp_path / "src"
        src = src_dir / "photo.jpg"
        _make_fake_jpeg(src)

        output_dir = tmp_path / "out"
        prefixes = {0: "c0_"}

        count = folders_export([src], prefixes, output_dir, dry_run=True)

        assert count == 0
        assert src.exists()
        assert not (output_dir / "c0" / "photo.jpg").exists()

    def test_collision_two_files_same_name_same_prefix(self, tmp_path):
        """Two files with same name and same prefix: second gets _1 suffix."""
        src_dir = tmp_path / "src"
        src1 = src_dir / "d1" / "duplicado.jpg"
        src2 = src_dir / "d2" / "duplicado.jpg"
        _make_fake_jpeg(src1)
        _make_fake_jpeg(src2)

        output_dir = tmp_path / "out"
        prefixes = {0: "c0_", 1: "c0_"}

        count = folders_export([src1, src2], prefixes, output_dir)

        assert count == 2
        dest1 = output_dir / "c0" / "duplicado.jpg"
        dest2 = output_dir / "c0" / "duplicado_1.jpg"
        assert dest1.exists()
        assert dest2.exists()
