#!/usr/bin/env python3
"""
Photo Clusterer — CLI tool for visual clustering using vision embeddings + UMAP + K-Means.

Clusters photos by visual similarity and generates an HTML gallery.

Pipeline:
  Photos ──► [DINOv2 | CLIP | SigLIP] ──► Embeddings ──► UMAP ──► K-Means ──► Folders + HTML

Usage:
    python cluster.py --input /path/to/photos --output /path/to/output [--preview]
"""

import argparse
import html
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cluster")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Cluster photos by visual similarity using vision embeddings + UMAP + K-Means",
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input directory containing photos"
    )
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


# ---------------------------------------------------------------------------
# Image scanner
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def scan_images(input_dir: str) -> list[Path]:
    """Recursively scan input_dir for image files."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        log.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(input_path.rglob(f"*{ext}"))
        files.extend(input_path.rglob(f"*{ext.upper()}"))

    files = sorted(set(files))
    if not files:
        log.error("No images found (jpg/jpeg/png/webp) in: %s", input_dir)
        sys.exit(1)
    log.info("Found %d images", len(files))
    return files


def sample_preview(files: list[Path], n: int = 10) -> list[Path]:
    return files if len(files) <= n else random.sample(files, n)


# ---------------------------------------------------------------------------
# Model backends
# ---------------------------------------------------------------------------

BACKENDS = {}  # lazy-import wrappers


def _load_dinov2(model_name: str, device: str):
    """Load DINOv2 model via HuggingFace transformers."""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    hf_name = {
        "dinov2-large": "facebook/dinov2-large",
        "dinov2-base": "facebook/dinov2-base",
    }[model_name]

    log.info("Loading DINOv2 model: %s ...", hf_name)
    t0 = time.time()

    processor = AutoImageProcessor.from_pretrained(hf_name)
    model = AutoModel.from_pretrained(hf_name).to(device)
    model.eval()

    log.info("DINOv2 loaded in %.1fs", time.time() - t0)
    return model, processor, device


def _load_clip(device: str):
    """Load CLIP ViT-B/32."""
    import torch
    import clip

    log.info("Loading CLIP ViT-B/32 ...")
    t0 = time.time()
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    log.info("CLIP loaded in %.1fs", time.time() - t0)
    return model, preprocess, device


def _load_siglip(device: str):
    """Load SigLIP via HuggingFace transformers."""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    hf_name = "google/siglip-base-patch16-224"
    log.info("Loading SigLIP model: %s ...", hf_name)
    t0 = time.time()

    processor = AutoImageProcessor.from_pretrained(hf_name)
    model = AutoModel.from_pretrained(hf_name).to(device)
    model.eval()

    log.info("SigLIP loaded in %.1fs", time.time() - t0)
    return model, processor, device


def load_model(model_name: str, device: str | None = None):
    """Load the selected model on GPU. Fails if CUDA is not available."""
    import torch

    if not torch.cuda.is_available():
        log.error("CUDA is not available — GPU is REQUIRED for this tool.")
        log.error("Run on a machine with an NVIDIA GPU and CUDA installed.")
        sys.exit(1)
    device = "cuda"
    log.info("Using device: %s", device)

    if model_name in ("dinov2-large", "dinov2-base"):
        model, processor, device = _load_dinov2(model_name, device)
        backend = "dinov2"
    elif model_name == "clip":
        model, processor, device = _load_clip(device)
        backend = "clip"
    elif model_name == "siglip":
        model, processor, device = _load_siglip(device)
        backend = "siglip"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, processor, backend, device


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# UMAP + K-Means clustering
# ---------------------------------------------------------------------------


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

    n_noise = 0  # K-Means assigns every point to a cluster
    log.info(
        "K-Means: k=%d (silhouette=%.3f), %d clusters",
        best_k,
        best_score if best_score > 0 else 0,
        len(set(best_labels)),
    )
    return best_labels


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------


def export_clusters(
    image_paths: list[Path],
    labels: np.ndarray,
    output_dir: Path,
    dry_run: bool = False,
) -> dict[int, list[tuple[int, Path]]]:
    """Organize images into cluster folders."""
    groups: dict[int, list[tuple[int, Path]]] = {}
    for idx, (label, path) in enumerate(zip(labels, image_paths)):
        groups.setdefault(int(label), []).append((idx, path))

    if dry_run:
        log.info("=== DRY RUN — no files will be written ===")
        for label in sorted(groups):
            items = groups[label]
            log.info("  cluster_%d/  → %d images", label, len(items))
        return groups

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, items in groups.items():
        dir_name = output_dir / f"cluster_{label}"
        dir_name.mkdir(exist_ok=True)

        for idx, src_path in items:
            dest = dir_name / src_path.name
            if dest.exists():
                dest = dir_name / f"{src_path.stem}_{idx}{src_path.suffix}"
            shutil.copy2(str(src_path), str(dest))

    log.info("Exported %d images to %s", len(image_paths), output_dir)
    return groups


# ---------------------------------------------------------------------------
# HTML gallery
# ---------------------------------------------------------------------------

HTML_HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Photo Clusters</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #111; color: #eee; min-height: 100vh; }
  .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
  h1 { font-size: 1.8rem; margin-bottom: 8px; }
  .subtitle { color: #888; margin-bottom: 24px; font-size: 0.9rem; }
  .cluster { margin-bottom: 40px; }
  .cluster h2 { font-size: 1.2rem; margin-bottom: 12px; color: #ccc;
                border-bottom: 1px solid #333; padding-bottom: 6px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
          gap: 8px; }
  .thumb { overflow: hidden; border-radius: 6px; background: #222;
           aspect-ratio: 1; position: relative; }
  .thumb img { width: 100%; height: 100%; object-fit: cover;
               transition: transform 0.2s; cursor: pointer; }
  .thumb img:hover { transform: scale(1.05); }
  .lb-overlay { display: none; position: fixed; inset: 0; z-index: 9999;
                background: rgba(0,0,0,0.92); justify-content: center;
                align-items: center; }
  .lb-overlay.active { display: flex; }
  .lb-overlay img { max-width: 90vw; max-height: 90vh; object-fit: contain;
                    border-radius: 4px; }
  .lb-close { position: absolute; top: 16px; right: 24px; color: #fff;
              font-size: 2rem; cursor: pointer; opacity: 0.7; }
  .lb-close:hover { opacity: 1; }
  .lb-prev, .lb-next { position: absolute; top: 50%; transform: translateY(-50%);
                        color: #fff; font-size: 3rem; cursor: pointer; opacity: 0.5;
                        padding: 16px; user-select: none; }
  .lb-prev:hover, .lb-next:hover { opacity: 1; }
  .lb-prev { left: 16px; }
  .lb-next { right: 16px; }
  .lb-counter { position: absolute; bottom: 16px; left: 50%; transform: translateX(-50%);
                color: #888; font-size: 0.85rem; }
  @media (max-width: 600px) {
    .grid { grid-template-columns: repeat(auto-fill, mincast(120px, 1fr)); }
  }
</style>
</head>
<body>
<div class="container">
<h1>Photo Clusters</h1>
<p class="subtitle">Clustered by visual similarity</p>
"""

HTML_FOOTER = """</div>

<div class="lb-overlay" id="lightbox">
  <span class="lb-close" onclick="closeLightbox()">&times;</span>
  <span class="lb-prev" onclick="navLightbox(-1)">&#10094;</span>
  <img id="lb-img" src="" alt="">
  <span class="lb-next" onclick="navLightbox(1)">&#10095;</span>
  <span class="lb-counter" id="lb-counter"></span>
</div>

<script>
let lbImages = [];
let lbIndex = 0;

document.querySelectorAll('[data-lightbox]').forEach(function(el, i) {
  el.addEventListener('click', function(e) {
    e.preventDefault();
    lbImages = Array.from(document.querySelectorAll('[data-lightbox]'));
    lbIndex = i;
    showLightbox();
  });
});

function showLightbox() {
  const el = lbImages[lbIndex];
  document.getElementById('lb-img').src = el.getAttribute('href');
  document.getElementById('lb-counter').textContent =
    (lbIndex + 1) + ' / ' + lbImages.length;
  document.getElementById('lightbox').classList.add('active');
}

function closeLightbox() {
  document.getElementById('lightbox').classList.remove('active');
}

function navLightbox(dir) {
  lbIndex = (lbIndex + dir + lbImages.length) % lbImages.length;
  showLightbox();
}

document.addEventListener('keydown', function(e) {
  if (!document.getElementById('lightbox').classList.contains('active')) return;
  if (e.key === 'Escape') closeLightbox();
  if (e.key === 'ArrowLeft') navLightbox(-1);
  if (e.key === 'ArrowRight') navLightbox(1);
});

document.getElementById('lightbox').addEventListener('click', function(e) {
  if (e.target === this) closeLightbox();
});
</script>
</body>
</html>"""


def generate_gallery(output_dir: Path, groups: dict[int, list[tuple[int, Path]]]):
    """Generate a self-contained index.html with inline CSS/JS."""
    output_dir = Path(output_dir)

    clusters = []
    for label in sorted(groups):
        items = groups[label]
        cluster_name = f"cluster_{label}"
        title = f"Cluster {label} ({len(items)} images)"

        thumbnails = []
        for idx, src_path in items:
            dest = f"{cluster_name}/{src_path.stem}{src_path.suffix}"
            if Path(output_dir / dest).exists():
                thumbnails.append((dest, src_path.name))
            elif Path(output_dir / f"{src_path.stem}_{idx}{src_path.suffix}").exists():
                dest = f"{cluster_name}/{src_path.stem}_{idx}{src_path.suffix}"
                thumbnails.append((dest, src_path.name))

        clusters.append(
            {
                "name": cluster_name,
                "title": title,
                "images": thumbnails,
            }
        )

    # Sort by size descending
    clusters.sort(key=lambda c: -len(c["images"]))

    html_parts = [HTML_HEADER]
    for cluster in clusters:
        html_parts.append(f'<div class="cluster" id="{html.escape(cluster["name"])}">')
        html_parts.append(f"  <h2>{html.escape(cluster['title'])}</h2>")
        html_parts.append('  <div class="grid">')
        for img_rel, img_name in cluster["images"]:
            escaped_rel = html.escape(img_rel, quote=True)
            escaped_name = html.escape(img_name, quote=True)
            html_parts.append(
                f'    <div class="thumb">'
                f'<a href="{escaped_rel}" data-lightbox="gallery" data-title="{escaped_name}">'
                f'<img src="{escaped_rel}" alt="{escaped_name}" loading="lazy">'
                f"</a></div>"
            )
        html_parts.append("  </div>")
        html_parts.append("</div>")

    html_parts.append(HTML_FOOTER)

    index_path = output_dir / "index.html"
    with open(index_path, "w") as f:
        f.write("\n".join(html_parts))
    log.info("Gallery: %s", index_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    output_dir = Path(args.output)

    # Phase 1: Scan
    images = scan_images(args.input)
    if args.preview:
        images = sample_preview(images, n=10)
        log.info("Preview mode: %d random images", len(images))

    # Phase 2: Load model
    model, processor, backend, device = load_model(args.model)

    # Phase 3: Embed
    embeddings = extract_embeddings(
        model,
        processor,
        backend,
        images,
        device,
        args.batch_size,
    )

    # Phase 4: Reduce + cluster (UMAP + K-Means)
    labels = reduce_and_cluster(embeddings, args.umap_components, args.max_clusters)

    # Report
    n_clusters = len(set(labels))
    log.info("Summary: %d clusters, %d total images", n_clusters, len(images))

    if args.dry_run:
        export_clusters(images, labels, output_dir, dry_run=True)
        return

    # Phase 5: Export
    groups = export_clusters(images, labels, output_dir, dry_run=False)

    # Phase 6: Gallery
    generate_gallery(output_dir, groups)

    log.info("Done! Gallery: %s", output_dir / "index.html")


if __name__ == "__main__":
    main()
