"""Model loading backends."""

import logging
import sys
import time

log = logging.getLogger("cluster")


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
