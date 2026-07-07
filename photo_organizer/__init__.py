"""Photo Organizer — visual clustering using vision embeddings + UMAP + K-Means."""

from photo_organizer.cli import parse_args
from photo_organizer.scanner import scan_images, sample_preview
from photo_organizer.models import load_model
from photo_organizer.embeddings import extract_embeddings
from photo_organizer.cluster import reduce_and_cluster
from photo_organizer.export import export_clusters
from photo_organizer.gallery import generate_gallery

__all__ = [
    "parse_args",
    "scan_images",
    "sample_preview",
    "load_model",
    "extract_embeddings",
    "reduce_and_cluster",
    "export_clusters",
    "generate_gallery",
]
