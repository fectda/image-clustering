"""UMAP dimensionality reduction + HDBSCAN clustering."""

import logging

import numpy as np

log = logging.getLogger("cluster")


def _build_prefix(label_path: list[int]) -> str:
    """Build a hierarchical prefix string from a list of cluster labels.

    Example: [5, 3] → "c5_c3_"
    Empty list → "" (noise at all levels).
    """
    if not label_path:
        return ""
    return "".join(f"c{label}_" for label in label_path)


def recursive_cluster(
    embeddings: np.ndarray,
    _image_paths: list | None = None,
    max_iterations: int = 3,
    min_cluster_size: int = 15,
) -> dict[int, str]:
    """Stack-based recursive clustering. Returns {image_index: prefix_string}.

    Prefix format: "c{label1}_c{label2}_" for depth-2, "c{label}_" for depth-1.
    Noise images (label=-1) at any depth get prefix "" (unclustered).
    """
    n_samples = len(embeddings)
    if n_samples == 0:
        return {}

    result: dict[int, str] = {}

    # Stack entries: (list_of_indices, current_depth, label_path_so_far)
    stack: list[tuple[list[int], int, list[int]]] = [(list(range(n_samples)), 0, [])]

    while stack:
        indices, depth, label_path = stack.pop()
        sub_embeddings = embeddings[indices]

        # Early stopping: skip if below min_cluster_size OR at max depth
        if len(indices) < min_cluster_size or depth >= max_iterations:
            prefix = _build_prefix(label_path)
            for idx in indices:
                result[idx] = prefix
            continue

        # Run clustering on sub-group
        labels, _ = reduce_and_cluster(sub_embeddings, min_cluster_size=min_cluster_size)

        # Group by cluster label
        clusters: dict[int, list[int]] = {}
        for i, label in enumerate(labels):
            orig_idx = indices[i]
            if label == -1:
                # Noise images get empty prefix at any depth
                result[orig_idx] = ""
            else:
                clusters.setdefault(label, []).append(orig_idx)

        # Push child groups onto stack
        for label, child_indices in clusters.items():
            stack.append((child_indices, depth + 1, label_path + [label]))

    return result


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

    # Early exit: need at least 2 samples for UMAP + HDBSCAN
    if n_samples < 2:
        log.warning("Only %d sample(s) — skipping clustering, all labeled as noise", n_samples)
        labels = np.full(n_samples, -1, dtype=int)
        metrics = {
            "dbcv": 0.0,
            "n_clusters": 0,
            "n_outliers": n_samples,
        }
        return labels, metrics

    # Clamp n_components to < n_samples (UMAP requires n_components < n_samples)
    orig_components = n_components
    n_components = min(n_components, max(1, n_samples - 1))
    if n_components != orig_components:
        log.info(
            "Adjusted n_components from %d to %d for %d samples",
            orig_components,
            n_components,
            n_samples,
        )

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
    # Defensive: unreachable via reduce_and_cluster (returns early), guards against direct calls
    if n_samples < 2:
        return 1
    return max(2, min(requested, n_samples - 1))
