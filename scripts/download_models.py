"""Pre-download vision models during Docker build so they're cached in the image.

This avoids downloading ~1.2 GB on every container run.
"""

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("download-models")

MODELS = [
    # Primary model
    "facebook/dinov2-large",
    # Secondary models (smaller, cached just in case)
    "facebook/dinov2-base",
    "openai/clip-vit-base-patch32",
    "google/siglip-base-patch16-224",
]


def main() -> int:
    # Import inside the script — not a module
    from transformers import AutoImageProcessor, AutoModel

    for model_id in MODELS:
        log.info(f"Downloading {model_id} ...")
        try:
            AutoModel.from_pretrained(model_id)
            AutoImageProcessor.from_pretrained(model_id)
            log.info(f"  ✅ {model_id}")
        except Exception as exc:
            log.warning(f"  ⚠️  {model_id} failed: {exc}")
            log.warning("  (continuing — only dinov2-large is required)")

    log.info("All models cached in image.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
