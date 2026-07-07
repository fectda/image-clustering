# syntax=docker/dockerfile:1

# ============================================================
# Stage 1: Builder — install deps + cache default model
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Default to CUDA 12.4. CPU-only torch is UNSUPPORTED — the tool requires GPU.
# Use `organize.sh` which handles GPU detection and builds with the correct arg.
# To override: --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ENV TORCH_INDEX_URL=${TORCH_INDEX_URL}

# Create virtual environment (portable — copy to runtime stage)
RUN python -m venv /venv
ENV PATH=/venv/bin:$PATH

# Install ALL Python deps in a single layer
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch>=2.2.0 \
    torchvision>=0.17.0 \
    --index-url ${TORCH_INDEX_URL} \
    --extra-index-url https://pypi.org/simple \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

# Pre-download only the default model (dinov2-large) so it's cached in the image.
# Other models download on first use when the user picks --model dinov2-base/clip/siglip.
ENV HF_HOME=/venv/hf-cache
RUN python <<EOF
from transformers import AutoImageProcessor, AutoModel
m = "facebook/dinov2-large"
AutoModel.from_pretrained(m)
AutoImageProcessor.from_pretrained(m)
print("✅", m)
EOF
RUN chmod -R a+rX /venv/hf-cache

# ============================================================
# Stage 2: Runtime — minimal production image
# ============================================================
FROM python:3.11-slim

WORKDIR /app

# System deps only — gosu for UID switching at runtime
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends gosu; \
    rm -rf /var/lib/apt/lists/*; \
    gosu nobody true

# Copy virtual environment with all Python deps + cached models
COPY --from=builder /venv /venv

# Application code
COPY photo_organizer/ /app/photo_organizer/
COPY scripts/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

ENV PATH=/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/venv/hf-cache

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "photo_organizer.main"]
