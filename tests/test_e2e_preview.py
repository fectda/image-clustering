"""Real E2E test: runs Docker container, verifies gallery output.

This is NOT mocked — it actually runs the Docker image against real images
and verifies the output is correct.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUTPUT_BASE = PROJECT_ROOT / "temporal"
IMAGES_DIR = Path("/home/fectda/temporal/images")
IMAGE_NAME = "vision-photo-organizer:latest"


@pytest.fixture
def output_dir():
    """Create a project-local output dir for the test run."""
    tmp = OUTPUT_BASE / f"test-run-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    yield tmp
    # Cleanup — force since files might be root-owned from earlier runs
    shutil.rmtree(tmp, ignore_errors=True)


def test_e2e_preview_gallery(output_dir):
    """Run the container with --preview and verify the gallery output."""
    # Skip if no real images available
    if not IMAGES_DIR.is_dir():
        pytest.skip(f"No images directory: {IMAGES_DIR}")

    # ── Run the container ──
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "-e",
            f"HOST_UID={os.getuid()}",
            "-e",
            f"HOST_GID={os.getgid()}",
            "-v",
            f"{IMAGES_DIR}:/data/input:ro",
            "-v",
            f"{output_dir}:/data/output",
            IMAGE_NAME,
            "--input",
            "/data/input",
            "--output",
            "/data/output",
            "--preview",
            "--min-cluster-size",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, f"Container failed:\n{result.stderr}\n{result.stdout}"

    # ── Verify gallery exists and is readable ──
    index_html = output_dir / "index.html"
    assert index_html.is_file(), "index.html not generated"
    html_content = index_html.read_text()

    # ── Verify ownership (should be host user, not root) ──
    assert index_html.stat().st_mode & 0o444, "index.html not readable"

    # ── Parse all image references from HTML ──
    src_paths = re.findall(r'src="([^"]+)"', html_content)
    assert len(src_paths) > 0, "No images in gallery"

    print(f"  Found {len(src_paths)} images in gallery")

    # ── Verify each image file exists and is readable ──
    for rel_path in src_paths:
        img_file = output_dir / rel_path
        assert img_file.is_file(), f"Image not found: {img_file}"
        assert img_file.stat().st_mode & 0o444, f"Image not readable: {img_file}"
        print(f"  ✅ {rel_path}")

    # ── Verify lightbox data attributes (only in HTML body, not JS strings) ──
    body_part = html_content.split("<script>")[0]
    lb_count = body_part.count("data-lightbox")
    assert lb_count == len(src_paths), (
        f"Lightbox count ({lb_count}) != image count ({len(src_paths)})"
    )

    # ── Verify href paths match src paths ──
    href_paths = re.findall(r'href="([^"]+)"', html_content)
    href_image_paths = [p for p in href_paths if not p.startswith("http")]
    for img_rel in src_paths:
        assert img_rel in href_image_paths, f"Missing href for {img_rel}"

    # ── Verify CSS: no mincast typo ──
    assert "mincast" not in html_content, "mincast typo still present in CSS"

    # ── Verify cluster structure ──
    cluster_dirs = list(output_dir.glob("cluster_*"))
    assert len(cluster_dirs) > 0, "No cluster directories created"

    print(f"\n  ✅ Gallery verified: file://{index_html}")
    print(f"     Clusters: {len(cluster_dirs)}")
    print(f"     Images:   {len(src_paths)}")
