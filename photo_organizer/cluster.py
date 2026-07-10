"""UMAP dimensionality reduction + HDBSCAN / KMeans clustering."""

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("cluster")

CLIP_CATEGORIES = [
    "A document, form, or receipt",
    "A screenshot of an app or screen",
    "A landscape or nature photo",
    "A photo of people or friends",
    "A photo of an animal or pet",
    "A photo of food or meal",
    "A photo of a vehicle or car",
]

CATEGORY_NAMES = [
    "document",
    "screenshot",
    "landscape",
    "people",
    "pet",
    "food",
    "vehicle",
]


def _build_prefix(label_path: list[int]) -> str:
    """Build a hierarchical prefix string from a list of cluster labels.

    Example: [5, 3] → "c5_c3_"
    Empty list → "" (noise at all levels).
    """
    if not label_path:
        return ""
    return "".join(f"c{label}_" for label in label_path)


def _cluster_kmeans(
    reduced: np.ndarray,
    k: int = 8,
    random_state: int = 42,
) -> np.ndarray:
    """Run KMeans on reduced embeddings. Returns flat labels array."""
    from sklearn.cluster import KMeans

    effective_k = min(k, max(2, len(reduced) - 1))
    log.info("Clustering with KMeans (k=%d) ...", effective_k)
    kmeans = KMeans(n_clusters=effective_k, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(reduced)
    return labels.astype(int)


def recursive_cluster(
    embeddings: np.ndarray,
    _image_paths: list[Path] | None = None,  # Reserved for future CLIP-fallback / noise-logging
    max_iterations: int = 3,
    min_cluster_size: int = 3,
    min_samples: int = 1,
    umap_n_components: int = 20,
    umap_n_neighbors: int = 40,
    umap_min_dist: float = 0.0,
    umap_metric: str = "cosine",
    cluster_algo: str = "kmeans",
    kmeans_k: int = 8,
) -> dict[int, str]:
    """Stack-based recursive clustering. Returns {image_index: prefix_string}.

    Both KMeans and HDBSCAN modes recurse via the same stack-based loop.
    KMeans noise handling is a no-op (KMeans never returns label -1).

    Prefix format: "c{label1}_c{label2}_" for depth-2, "c{label}_" for depth-1.
    Noise images (label=-1) at any depth get prefix "" (unclustered).
    """
    n_samples = len(embeddings)
    if n_samples == 0:
        return {}

    result: dict[int, str] = {}

    # Stack-based recursive clustering
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
        labels, _ = reduce_and_cluster(
            sub_embeddings,
            n_components=umap_n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            algo=cluster_algo,
            kmeans_k=kmeans_k,
        )

        # Group by cluster label
        clusters: dict[int, list[int]] = {}
        for i, label in enumerate(labels):
            orig_idx = indices[i]
            if label == -1:
                # Noise images retain their parent's prefix and stop recursing
                result[orig_idx] = _build_prefix(label_path)
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
    min_cluster_size: int = 3,
    min_samples: int = 1,
    algo: str = "hdbscan",
    kmeans_k: int = 8,
) -> tuple[np.ndarray, dict]:
    """Reduce dims with UMAP, then cluster with HDBSCAN or KMeans.

    Returns (labels, metrics_dict) where metrics = {dbcv, n_clusters, n_outliers}.
    For KMeans, dbcv is always 0.0 and n_outliers is always 0.
    """
    import umap

    n_samples = len(embeddings)

    # Early exit: need at least 3 samples for UMAP (n_neighbors < n_samples strict)
    if n_samples < 3:
        log.warning("Only %d sample(s) — skipping clustering, all labeled as noise", n_samples)
        labels = np.full(n_samples, -1, dtype=int)
        metrics = {
            "dbcv": 0.0,
            "n_clusters": 0,
            "n_outliers": n_samples,
        }
        return labels, metrics

    # Clamp n_components to < n_samples - 1 to avoid scipy eigsh error (k >= N)
    orig_components = n_components
    n_components = min(n_components, max(1, n_samples - 2))
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

    # Dispatch to algorithm
    if algo == "kmeans":
        labels = _cluster_kmeans(reduced, k=kmeans_k)
        n_clusters = len(set(labels))
        metrics = {
            "dbcv": 0.0,
            "n_clusters": n_clusters,
            "n_outliers": 0,
        }
        log.info(
            "KMeans clustering complete: %d clusters",
            n_clusters,
        )
        # Log cluster size distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            log.info("  cluster_%d: %d images", label, count)
        return labels, metrics

    # HDBSCAN clustering
    import hdbscan

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

        dbcv_score = float(validity_index(reduced.astype(np.float64), labels))

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
    for label, count in zip(unique_labels, counts):
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


def classify_noise_with_clip(
    image_paths: list[Path],
    prefixes: dict[int, str],
    device: str = "cpu",
) -> None:
    """Classify noise images (empty prefix) using CLIP zero-shot.

    Modifies prefixes dict in-place, updating noise-image entries to
    "c_misc_[category]_" prefixes.

    Parameters
    ----------
    image_paths : list[Path]
        All image paths (same order as prefixes keys).
    prefixes : dict[int, str]
        Current prefix map from recursive_cluster(). Modified in-place.
    device : str
        Device for CLIP inference.

    Side Effects
    ------------
    - Loads CLIP model via transformers (lazy import)
    - Logs classification results
    - Frees model after classification
    """
    # Collect noise image indices (empty prefix)
    noise_indices = [idx for idx, prefix in prefixes.items() if prefix == ""]
    if not noise_indices:
        log.info("No noise images to classify with CLIP — skipping.")
        return None

    log.info("Classifying %d noise images with CLIP zero-shot ...", len(noise_indices))

    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        log.warning(
            "Required dependencies (torch, transformers, Pillow) not available for CLIP fallback"
        )
        return None

    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception as exc:
        log.warning("Failed to load CLIP model (%s) — skipping CLIP fallback.", exc)
        return None

    model = model.to(device)
    model.eval()

    # Filter to only noise image paths
    noise_paths = [image_paths[idx] for idx in noise_indices]

    # Load and preprocess images
    pil_images = []
    valid_indices = []
    for i, path in enumerate(noise_paths):
        try:
            img = Image.open(path).convert("RGB")
            pil_images.append(img)
            valid_indices.append(noise_indices[i])
        except Exception as exc:
            log.warning("Skipping corrupt image: %s (%s)", path, exc)

    if not pil_images:
        log.warning("No valid noise images to classify.")
        return None

    # Encode images and text prompts
    with torch.no_grad():
        image_inputs = processor(images=pil_images, return_tensors="pt", padding=True).to(device)
        text_inputs = processor(text=CLIP_CATEGORIES, return_tensors="pt", padding=True).to(device)

        image_embeds = model.get_image_features(**image_inputs)
        text_embeds = model.get_text_features(**text_inputs)

        # L2-normalize
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # Cosine similarity → argmax
        similarity = (image_embeds @ text_embeds.T).softmax(dim=-1)
        best_category_idx = similarity.argmax(dim=-1)

    # Update prefixes in-place
    for i, category_idx in enumerate(best_category_idx):
        idx = valid_indices[i]
        category_name = CATEGORY_NAMES[category_idx.item()]
        prefixes[idx] = f"c_misc_{category_name}_"

    # Free model
    del model
    del processor
    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info(
        "CLIP classification complete: %d images classified.",
        len(valid_indices),
    )

    return None
