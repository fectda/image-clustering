"""Photo Organizer — CLI tool for visual clustering using vision embeddings + UMAP + K-Means."""

import logging
import sys
from pathlib import Path

from photo_organizer.cli import parse_args
from photo_organizer.scanner import scan_images, sample_preview
from photo_organizer.models import load_model
from photo_organizer.embeddings import extract_embeddings
from photo_organizer.cluster import reduce_and_cluster
from photo_organizer.export import export_clusters
from photo_organizer.gallery import generate_gallery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cluster")


def main():
    args = parse_args()
    output_dir = Path(args.output)

    # Phase 1: Scan
    images = scan_images(args.input)
    if args.preview:
        images = sample_preview(images, n=10)
        log.info("Preview mode: %d random images", len(images))

    # Phase 2: Load model
    model, processor, backend, device = load_model(args.model)

    # Phase 3: Embed
    embeddings = extract_embeddings(
        model,
        processor,
        backend,
        images,
        device,
        args.batch_size,
    )

    # Phase 4: Reduce + cluster (UMAP + K-Means)
    labels = reduce_and_cluster(embeddings, args.umap_components, args.max_clusters)

    # Report
    n_clusters = len(set(labels))
    log.info("Summary: %d clusters, %d total images", n_clusters, len(images))

    if args.dry_run:
        export_clusters(images, labels, output_dir, dry_run=True)
        return

    # Phase 5: Export
    groups = export_clusters(images, labels, output_dir, dry_run=False)

    # Phase 6: Gallery
    generate_gallery(output_dir, groups)

    log.info("Done! Gallery: %s", output_dir / "index.html")


if __name__ == "__main__":
    main()
