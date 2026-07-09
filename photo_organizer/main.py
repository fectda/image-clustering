"""Photo Organizer — CLI tool for visual clustering using vision embeddings + UMAP + HDBSCAN."""

import logging
import sys
from pathlib import Path

from photo_organizer.cli import parse_args
from photo_organizer.cluster import classify_noise_with_clip, recursive_cluster
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
    model: str = "dinov2-large",
    batch_size: int = 32,
    max_iterations: int = 3,
    min_cluster_size: int = 3,
    min_samples: int = 1,
    clip_fallback: bool = True,
    dry_run: bool = False,
    preview: bool = False,
    umap_n_components: int = 20,
    umap_n_neighbors: int = 40,
    umap_min_dist: float = 0.0,
    umap_metric: str = "cosine",
) -> None:
    """Full pipeline: scan → embed → recursive_cluster → [clip_fallback] → export → gallery."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Phase 1: Scan
    images = scan_images(input_dir)
    if preview:
        images = sample_preview(images, n=10)
        log.info("Preview mode: %d random images", len(images))

    # Phase 2: Load model
    model, processor, backend, device = load_model(model)

    # Phase 3: Embed
    if backend == "hybrid":
        embeddings, images = extract_hybrid_embeddings(
            model,
            images,
            device,
            batch_size,
        )
    else:
        embeddings, images = extract_embeddings(
            model,
            processor,
            backend,
            images,
            device,
            batch_size,
        )

    # Phase 4: Recursive clustering
    prefixes = recursive_cluster(
        embeddings,
        images,
        max_iterations=max_iterations,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        umap_n_components=umap_n_components,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        umap_metric=umap_metric,
    )

    # Phase 4b: CLIP fallback for residual noise images
    if clip_fallback:
        if backend == "clip":
            log.info("Primary model is already CLIP-based, skipping CLIP fallback.")
        else:
            noise_images = [images[idx] for idx, p in prefixes.items() if p == ""]
            if noise_images:
                log.info(
                    "Running CLIP fallback on %d noise images.",
                    len(noise_images),
                )
                classify_noise_with_clip(images, prefixes, device=device)
            else:
                log.info("No residual noise images — skipping CLIP fallback.")

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
            "Use --min-cluster-size instead (default: 3)."
        )

    organize_photos(
        args.input,
        args.output,
        model=args.model,
        batch_size=args.batch_size,
        max_iterations=args.max_iterations,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        clip_fallback=args.clip_fallback,
        dry_run=args.dry_run,
        preview=args.preview,
        umap_n_components=args.umap_components,
        umap_n_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_metric=args.umap_metric,
    )


if __name__ == "__main__":
    main()
