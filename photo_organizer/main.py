"""Photo Organizer — CLI tool for visual clustering using vision embeddings + UMAP + HDBSCAN."""

import logging
import sys
from pathlib import Path

from photo_organizer.cli import parse_args
from photo_organizer.cluster import reduce_and_cluster
from photo_organizer.embeddings import extract_embeddings, extract_hybrid_embeddings
from photo_organizer.export import export_clusters
from photo_organizer.gallery import generate_gallery
from photo_organizer.models import load_model
from photo_organizer.scanner import sample_preview, scan_images

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cluster")


def main():
    args = parse_args()
    output_dir = Path(args.output)

    # Deprecation warning for --max-clusters
    if "--max-clusters" in sys.argv:
        log.warning(
            "DEPRECATED: --max-clusters is ignored by HDBSCAN. "
            "Use --min-cluster-size instead (default: 15)."
        )

    # Phase 1: Scan
    images = scan_images(args.input)
    if args.preview:
        images = sample_preview(images, n=10)
        log.info("Preview mode: %d random images", len(images))

    # Phase 2: Load model
    model, processor, backend, device = load_model(args.model)

    # Phase 3: Embed
    if backend == "hybrid":
        embeddings = extract_hybrid_embeddings(
            model,
            images,
            device,
            args.batch_size,
        )
    else:
        embeddings = extract_embeddings(
            model,
            processor,
            backend,
            images,
            device,
            args.batch_size,
        )

    # Phase 4: Reduce + cluster (UMAP + HDBSCAN)
    labels, metrics = reduce_and_cluster(
        embeddings,
        n_components=args.umap_components,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )

    # Report
    log.info(
        "Summary: %d clusters, %d outliers, DBCV=%.3f, %d total images",
        metrics["n_clusters"],
        metrics["n_outliers"],
        metrics["dbcv"],
        len(images),
    )

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
