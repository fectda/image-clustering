"""CLI argument parsing."""

import argparse


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Cluster photos by visual similarity using vision embeddings + UMAP + K-Means",
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
        choices=["dinov2-large", "dinov2-base", "clip", "siglip"],
        help="Vision model for embeddings (default: dinov2-large)",
    )
    parser.add_argument(
        "--umap-components",
        type=int,
        default=5,
        help="UMAP target dimensions (default: 5)",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=20,
        help="Maximum K clusters for silhouette search (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding extraction (default: 32)",
    )
    return parser.parse_args(argv)
