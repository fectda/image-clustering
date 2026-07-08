"""Model loading backends."""

import logging
import sys
import time

log = logging.getLogger("cluster")


def _load_dinov2(model_name: str, device: str):
    """Load DINOv2 model via HuggingFace transformers."""
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
    import clip

    log.info("Loading CLIP ViT-B/32 ...")
    t0 = time.time()
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    log.info("CLIP loaded in %.1fs", time.time() - t0)
    return model, preprocess, device


def _load_siglip(device: str):
    """Load SigLIP via HuggingFace transformers."""
    from transformers import AutoImageProcessor, AutoModel

    hf_name = "google/siglip-base-patch16-224"
    log.info("Loading SigLIP model: %s ...", hf_name)
    t0 = time.time()

    processor = AutoImageProcessor.from_pretrained(hf_name)
    model = AutoModel.from_pretrained(hf_name).to(device)
    model.eval()

    log.info("SigLIP loaded in %.1fs", time.time() - t0)
    return model, processor, device


def _load_qwen3(device: str):
    """Load Qwen3-VL-Embedding-2B via HuggingFace transformers.

    Tries AutoImageProcessor first; falls back to AutoProcessor if unavailable.
    """
    from transformers import AutoModel

    hf_name = "Qwen/Qwen3-VL-Embedding-2B"
    log.info("Loading Qwen3 model: %s ...", hf_name)
    t0 = time.time()

    # Try AutoImageProcessor first, fall back to AutoProcessor
    try:
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(hf_name)
    except (ImportError, OSError):
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(hf_name)

    model = AutoModel.from_pretrained(hf_name).to(device)
    model.eval()

    log.info("Qwen3 loaded in %.1fs", time.time() - t0)
    return model, processor, device


def _load_hybrid(device: str):
    """Load both DINOv2-large and Qwen3-VL-Embedding-2B for hybrid embeddings.

    Returns a dict with both models for sequential embedding extraction.
    """
    log.info("Loading hybrid model (DINOv2-large + Qwen3-2B) ...")
    t0 = time.time()

    dinov2_model, dinov2_proc, _ = _load_dinov2("dinov2-large", device)
    qwen3_model, qwen3_proc, _ = _load_qwen3(device)

    log.info("Hybrid models loaded in %.1fs", time.time() - t0)
    return (
        {
            "dinov2": (dinov2_model, dinov2_proc),
            "qwen3": (qwen3_model, qwen3_proc),
        },
        None,  # no single processor
        "hybrid",
        device,
    )


def load_model(model_name: str, device: str | None = None):
    """Load the selected model on GPU. Fails if CUDA is not available."""
    import torch

    if not torch.cuda.is_available():
        log.error("=" * 70)
        log.error("CUDA is not available — GPU is REQUIRED for this tool.")
        log.error("=" * 70)
        log.error("Possible causes:")
        log.error("  1. Image was built without CUDA support")
        log.error(
            "     → Rebuild with: --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124"
        )
        log.error("  2. Container running without GPU access")
        log.error("     → Add --gpus all to docker run, or use ./organize.sh")
        log.error("  3. NVIDIA Container Toolkit not installed on the host")
        log.error("     → https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/")
        log.error("  4. No NVIDIA GPU on the host machine")
        log.error("=" * 70)
        sys.exit(1)
    device = device or "cuda"
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
    elif model_name == "hybrid":
        model, processor, backend, device = _load_hybrid(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, processor, backend, device
