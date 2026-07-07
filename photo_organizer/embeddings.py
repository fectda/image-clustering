"""Embedding extraction from images."""

import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

log = logging.getLogger("cluster")


def extract_embeddings(
    model,
    processor,
    backend: str,
    image_paths: list[Path],
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract L2-normalized embeddings for all images. Returns (N, D) array."""
    import torch

    log.info(
        "Extracting embeddings for %d images (backend=%s) ...",
        len(image_paths),
        backend,
    )
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding"):
            batch_paths = image_paths[i : i + batch_size]
            images = []
            valid_indices = []

            for idx, img_path in enumerate(batch_paths):
                try:
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                    valid_indices.append(idx)
                except Exception as exc:
                    log.warning("Skipping corrupt image: %s (%s)", img_path, exc)

            if not images:
                continue

            if backend == "clip":
                # CLIP uses its own preprocess pipeline
                batch_input = torch.stack([processor(img) for img in images]).to(device)
                features = model.encode_image(batch_input)
            elif backend in ("dinov2", "siglip"):
                # HF transformers: processor returns a dict
                inputs = processor(images=images, return_tensors="pt").to(device)
                outputs = model(**inputs)
                # Use CLS token (first token) for classification/embedding
                if backend == "siglip":
                    # SigLIP: mean pool the vision tokens
                    features = outputs.pooler_output
                else:
                    # DINOv2: use CLS token
                    features = outputs.last_hidden_state[:, 0, :]
            else:
                raise ValueError(f"Unknown backend: {backend}")

            features = features / features.norm(dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy())

    if not all_embeddings:
        log.error("No valid images could be processed.")
        sys.exit(1)

    embeddings = np.concatenate(all_embeddings, axis=0)
    log.info("Embedding shape: %s", embeddings.shape)
    return embeddings
