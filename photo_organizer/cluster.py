"""UMAP dimensionality reduction + K-Means clustering."""

import logging

import numpy as np

log = logging.getLogger("cluster")


def reduce_and_cluster(
    embeddings: np.ndarray,
    n_components: int = 5,
    max_k: int = 20,
) -> np.ndarray:
    """Reduce dims with UMAP, then cluster with K-Means (silhouette-optimized K)."""
    log.info("Reducing to %d dims with UMAP ...", n_components)

    import umap

    reducer = umap.UMAP(
        n_components=n_components,
        random_state=42,
        n_neighbors=min(15, len(embeddings) - 1),
        min_dist=0.1,
    )
    reduced = reducer.fit_transform(embeddings)
    log.info("UMAP reduced shape: %s", reduced.shape)

    # Auto-detect K via silhouette score
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n_samples = len(reduced)
    max_k = min(max_k, n_samples - 1)

    best_score = -1
    best_k = min(5, max_k)  # default fallback
    best_labels = None

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(reduced)
        if len(set(labels)) > 1:
            score = silhouette_score(reduced, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

    if best_labels is None:
        # Edge case: fallback to 2 clusters
        km = KMeans(n_clusters=2, random_state=42, n_init="auto")
        best_labels = km.fit_predict(reduced)

    log.info(
        "K-Means: k=%d (silhouette=%.3f), %d clusters",
        best_k,
        best_score if best_score > 0 else 0,
        len(set(best_labels)),
    )
    return best_labels
