# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
# For CUDA support, build with: --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV TORCH_INDEX_URL=${TORCH_INDEX_URL}

# Install torch + torchvision from the pytorch index, other deps from PyPI
RUN pip install --no-cache-dir \
    torch>=2.2.0 \
    torchvision>=0.17.0 \
    --index-url ${TORCH_INDEX_URL} \
    --extra-index-url https://pypi.org/simple

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download vision models into the image (so they don't download on every run)
# Store in world-readable location so host-UID switching works at runtime
ENV HF_HOME=/usr/local/share/hf-cache
COPY scripts/download_models.py /tmp/
RUN python /tmp/download_models.py && \
    rm /tmp/download_models.py && \
    chmod -R a+rX /usr/local/share/hf-cache

# Install gosu for dropping privileges in entrypoint
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends gosu; \
    rm -rf /var/lib/apt/lists/*; \
    gosu nobody true

# Copy application
COPY photo_organizer/ /app/photo_organizer/
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "photo_organizer.main"]
