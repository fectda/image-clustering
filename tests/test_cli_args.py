"""Tests for CLI argument parsing — new flags: --no-gallery, --no-recursive-search, --output-mode."""

import pytest

from photo_organizer.cli import parse_args


class TestNoGalleryFlag:
    """--no-gallery flag parsing."""

    def test_no_gallery_flag_true(self):
        args = parse_args(["--input", "/in", "--output", "/out", "--no-gallery"])
        assert args.no_gallery is True

    def test_no_gallery_default_false(self):
        args = parse_args(["--input", "/in", "--output", "/out"])
        assert args.no_gallery is False


class TestNoRecursiveSearchFlag:
    """--no-recursive-search flag parsing."""

    def test_no_recursive_search_flag_true(self):
        args = parse_args(["--input", "/in", "--output", "/out", "--no-recursive-search"])
        assert args.no_recursive_search is True

    def test_no_recursive_search_default_false(self):
        args = parse_args(["--input", "/in", "--output", "/out"])
        assert args.no_recursive_search is False


class TestOutputModeFlag:
    """--output-mode flag parsing."""

    def test_output_mode_folders(self):
        args = parse_args(["--input", "/in", "--output", "/out", "--output-mode", "folders"])
        assert args.output_mode == "folders"

    def test_output_mode_flat_explicit(self):
        args = parse_args(["--input", "/in", "--output", "/out", "--output-mode", "flat"])
        assert args.output_mode == "flat"

    def test_output_mode_default_flat(self):
        args = parse_args(["--input", "/in", "--output", "/out"])
        assert args.output_mode == "flat"

    def test_output_mode_invalid(self):
        with pytest.raises(SystemExit):
            parse_args(["--input", "/in", "--output", "/out", "--output-mode", "invalid"])


class TestCombinedFlags:
    """All new flags used together."""

    def test_all_flags_combined(self):
        args = parse_args(
            [
                "--input",
                "/in",
                "--output",
                "/out",
                "--no-gallery",
                "--no-recursive-search",
                "--output-mode",
                "folders",
            ]
        )
        assert args.no_gallery is True
        assert args.no_recursive_search is True
        assert args.output_mode == "folders"

    def test_defaults_combined(self):
        args = parse_args(["--input", "/in", "--output", "/out"])
        assert args.no_gallery is False
        assert args.no_recursive_search is False
        assert args.output_mode == "flat"
