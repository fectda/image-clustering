"""UMAP dimensionality reduction + HDBSCAN clustering."""

import logging

import numpy as np

log = logging.getLogger("cluster")


def reduce_and_cluster(
    embeddings: np.ndarray,
    n_components: int = 20,
    n_neighbors: int = 40,
    min_dist: float = 0.0,
    metric: str = "cosine",
    min_cluster_size: int = 15,
    min_samples: int = 5,
) -> tuple[np.ndarray, dict]:
    """Reduce dims with UMAP, then cluster with HDBSCAN.

    Returns (labels, metrics_dict) where metrics = {dbcv, n_clusters, n_outliers}.
    """
    import hdbscan
    import umap

    n_samples = len(embeddings)

    # UMAP dimensionality reduction
    effective_neighbors = min_neighbors(n_neighbors, n_samples)
    log.info(
        "Reducing to %d dims with UMAP (neighbors=%d, dist=%.2f, metric=%s) ...",
        n_components,
        effective_neighbors,
        min_dist,
        metric,
    )

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    reduced = reducer.fit_transform(embeddings)
    log.info("UMAP reduced shape: %s", reduced.shape)

    # HDBSCAN clustering
    log.info(
        "Clustering with HDBSCAN (min_cluster_size=%d, min_samples=%d) ...",
        min_cluster_size,
        min_samples,
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
        metric="euclidean",
    )
    labels = clusterer.fit_predict(reduced)

    # Compute metrics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = int(np.sum(labels == -1))

    # DBCV score
    if n_clusters < 2 or n_outliers == n_samples:
        # All noise or single cluster — DBCV undefined
        dbcv_score = 0.0
        if n_outliers == n_samples:
            log.warning(
                "All %d images classified as noise. "
                "Consider lowering --min-cluster-size (currently %d).",
                n_samples,
                min_cluster_size,
            )
    else:
        from hdbscan.validity import validity_index

        dbcv_score = float(validity_index(reduced, labels))

    metrics = {
        "dbcv": dbcv_score,
        "n_clusters": n_clusters,
        "n_outliers": n_outliers,
    }

    # Log results
    log.info(
        "Clustering complete: %d clusters, %d outliers, DBCV=%.3f",
        n_clusters,
        n_outliers,
        dbcv_score,
    )

    # Log cluster size distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts, strict=False):
        if label == -1:
            log.info("  noise: %d images", count)
        else:
            log.info("  cluster_%d: %d images", label, count)

    return labels, metrics


def min_neighbors(requested: int, n_samples: int) -> int:
    """Ensure n_neighbors < n_samples and at least 2."""
    return max(2, min(requested, max(n_samples - 1, 2)))
