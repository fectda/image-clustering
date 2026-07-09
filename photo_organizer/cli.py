"""CLI argument parsing."""

import argparse


def positive_int(v: str) -> int:
    """Convert to int and validate >= 1."""
    iv = int(v)
    if iv < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {iv}")
    return iv


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Cluster photos by visual similarity using vision embeddings + UMAP + clustering",
    )
    parser.add_argument("--input", "-i", required=True, help="Input directory containing photos")
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory for clustered photos + gallery",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Process a random sample of 10 photos for quick validation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report without writing any files",
    )
    parser.add_argument(
        "--model",
        default="dinov2-large",
        choices=["dinov2-large", "dinov2-base", "clip", "siglip", "hybrid"],
        help="Vision model for embeddings (default: dinov2-large)",
    )
    parser.add_argument(
        "--umap-components",
        type=int,
        default=20,
        help="UMAP target dimensions (default: 20)",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=40,
        help="UMAP n_neighbors (default: 40)",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.0,
        help="UMAP min_dist (default: 0.0)",
    )
    parser.add_argument(
        "--umap-metric",
        default="cosine",
        choices=["cosine", "euclidean"],
        help="UMAP distance metric (default: cosine)",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=20,
        help="DEPRECATED: Maximum K clusters for silhouette search. Ignored by HDBSCAN (default: 20)",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=3,
        help="HDBSCAN minimum cluster size (default: 3)",
    )
    parser.add_argument(
        "--min-samples",
        type=positive_int,
        default=1,
        help="HDBSCAN minimum samples (default: 1)",
    )
    parser.add_argument(
        "--clip-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CLIP zero-shot fallback for noise images (default: true)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding extraction (default: 32)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum recursive clustering depth (default: 3)",
    )
    return parser.parse_args(argv)
