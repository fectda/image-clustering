"""Embedding extraction from images."""

import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

log = logging.getLogger("cluster")


def _extract_single_backend(
    model,
    processor,
    backend: str,
    image_paths: list[Path],
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, list[Path]]:
    """Extract embeddings from a single backend.

    Returns
    -------
    (N, D) L2-normalized array and filtered list of valid image paths.
    """
    import torch

    if batch_size < 1:
        log.warning("batch_size=%d is invalid, using batch_size=1", batch_size)
        batch_size = 1

    all_embeddings = []
    valid_paths: list[Path] = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc=f"Embedding ({backend})"):
            batch_paths = image_paths[i : i + batch_size]
            images = []
            batch_valid: list[Path] = []

            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                    batch_valid.append(img_path)
                except Exception as exc:
                    log.warning("Skipping corrupt image: %s (%s)", img_path, exc)

            if not images:
                continue

            if backend == "clip":
                batch_input = torch.stack([processor(img) for img in images]).to(device)
                features = model.encode_image(batch_input)
            elif backend in ("dinov2", "siglip"):
                inputs = processor(images=images, return_tensors="pt").to(device)
                outputs = model(**inputs)
                if backend == "siglip":
                    features = outputs.pooler_output
                else:
                    features = outputs.last_hidden_state[:, 0, :]
            elif backend == "qwen3":
                inputs = processor(images=images, return_tensors="pt").to(device)
                outputs = model(**inputs)
                # Qwen3-VL-Embedding-2B: use the pooled embedding
                features = outputs.pooler_output
            else:
                raise ValueError(f"Unknown backend: {backend}")

            features = features / features.norm(dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy())
            valid_paths.extend(batch_valid)

    if not all_embeddings:
        return np.array([]), []

    return np.concatenate(all_embeddings, axis=0), valid_paths


def extract_hybrid_embeddings(
    models: dict,
    image_paths: list[Path],
    device: str,
    batch_size: int = 32,
) -> tuple[np.ndarray, list[Path]]:
    """Extract hybrid embeddings: DINOv2 (1024d) + Qwen3 (4096d) → (N, 5120).

    Each sub-model's output is L2-normalized independently before concatenation.
    The combined embedding is L2-normalized after concatenation.

    Returns
    -------
    (N, 5120) L2-normalized array and filtered list of valid image paths.
    """
    log.info(
        "Extracting hybrid embeddings for %d images (DINOv2 + Qwen3) ...",
        len(image_paths),
    )

    # Extract DINOv2 embeddings (1024d)
    dinov2_model, dinov2_proc = models["dinov2"]
    dinov2_emb, valid_paths = _extract_single_backend(
        dinov2_model, dinov2_proc, "dinov2", image_paths, device, batch_size
    )
    if dinov2_emb.size == 0:
        log.error("No valid images for DINOv2 embedding.")
        sys.exit(1)
    log.info("DINOv2 embeddings: %s", dinov2_emb.shape)

    # Extract Qwen3 embeddings (4096d) — may need smaller batch size
    qwen3_batch = min(batch_size, 16)
    qwen3_model, qwen3_proc = models["qwen3"]
    qwen3_emb, qwen3_paths = _extract_single_backend(
        qwen3_model, qwen3_proc, "qwen3", image_paths, device, qwen3_batch
    )
    if qwen3_emb.size == 0:
        log.error("No valid images for Qwen3 embedding.")
        sys.exit(1)
    log.info("Qwen3 embeddings: %s", qwen3_emb.shape)

    # Verify image count matches
    if dinov2_emb.shape[0] != qwen3_emb.shape[0]:
        log.error(
            "Image count mismatch: DINOv2=%d, Qwen3=%d",
            dinov2_emb.shape[0],
            qwen3_emb.shape[0],
        )
        sys.exit(1)

    # Verify both backends processed the same paths
    if valid_paths != qwen3_paths:
        log.error(
            "Path mismatch between DINOv2 and Qwen3 (%d vs %d). "
            "Both backends must process the same images.",
            len(valid_paths),
            len(qwen3_paths),
        )
        sys.exit(1)

    # Concatenate → L2-normalize
    combined = np.concatenate([dinov2_emb, qwen3_emb], axis=1)
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    combined = combined / norms

    log.info("Hybrid embedding shape: %s", combined.shape)
    return combined, valid_paths


def extract_embeddings(
    model,
    processor,
    backend: str,
    image_paths: list[Path],
    device: str,
    batch_size: int = 32,
) -> tuple[np.ndarray, list[Path]]:
    """Extract L2-normalized embeddings for all images.

    Returns
    -------
    (N, D) L2-normalized array and filtered list of valid image paths.
    """
    log.info(
        "Extracting embeddings for %d images (backend=%s) ...",
        len(image_paths),
        backend,
    )

    embeddings, valid_paths = _extract_single_backend(
        model, processor, backend, image_paths, device, batch_size
    )

    if embeddings.size == 0:
        log.error("No valid images could be processed.")
        sys.exit(1)

    log.info("Embedding shape: %s", embeddings.shape)
    return embeddings, valid_paths
