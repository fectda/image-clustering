"""Photo Organizer — CLI tool for visual clustering using vision embeddings + UMAP + HDBSCAN."""

import logging
import sys
from pathlib import Path

from photo_organizer.cli import parse_args
from photo_organizer.cluster import recursive_cluster
from photo_organizer.embeddings import extract_embeddings, extract_hybrid_embeddings
from photo_organizer.export import flat_export
from photo_organizer.gallery import generate_flat_gallery
from photo_organizer.models import load_model
from photo_organizer.scanner import sample_preview, scan_images

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cluster")


def organize_photos(
    input_dir,
    output_dir,
    max_iterations: int = 3,
    dry_run: bool = False,
    preview: bool = False,
) -> None:
    """Full pipeline: scan → embed (once) → recursive_cluster → flat_export → flat_gallery.

    Standalone API that runs the full pipeline without requiring a CLI args namespace.
    Uses hardcoded defaults for model, batch_size, min_cluster_size.
    Supports dry_run and preview modes.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Phase 1: Scan
    images = scan_images(input_dir)
    if preview:
        images = sample_preview(images, n=10)
        log.info("Preview mode: %d random images", len(images))

    # Phase 2: Load model (hardcoded default)
    model, processor, backend, device = load_model("dinov2-large")

    # Phase 3: Embed (hardcoded batch_size)
    if backend == "hybrid":
        embeddings, images = extract_hybrid_embeddings(
            model,
            images,
            device,
            32,
        )
    else:
        embeddings, images = extract_embeddings(
            model,
            processor,
            backend,
            images,
            device,
            32,
        )

    # Phase 4: Recursive clustering (hardcoded min_cluster_size)
    prefixes = recursive_cluster(
        embeddings,
        images,
        max_iterations=max_iterations,
        min_cluster_size=15,
    )

    # Phase 5: Export
    if dry_run:
        flat_export(images, prefixes, output_dir, dry_run=True)
        return

    flat_export(images, prefixes, output_dir, dry_run=False)

    # Phase 6: Gallery
    generate_flat_gallery(output_dir)

    log.info("Done! Gallery: %s", output_dir / "index.html")


def main():
    args = parse_args()

    # Deprecation warning for --max-clusters
    if "--max-clusters" in sys.argv:
        log.warning(
            "DEPRECATED: --max-clusters is ignored by HDBSCAN. "
            "Use --min-cluster-size instead (default: 15)."
        )

    organize_photos(
        args.input,
        args.output,
        max_iterations=args.max_iterations,
        dry_run=args.dry_run,
        preview=args.preview,
    )


if __name__ == "__main__":
    main()
