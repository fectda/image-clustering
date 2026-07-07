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

# Copy application
COPY cluster.py .

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "/app/cluster.py"]
