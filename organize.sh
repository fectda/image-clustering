#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────
# Photo Organizer — Docker wrapper
# ─────────────────────────────────────────────────────────────
# Usage:
#   ./organize.sh --input /path/to/photos --output /path/to/out [OPTIONS]
#
# Options:
#   --input, -i       Input directory (required)
#   --output, -o      Output directory (required)
#   --preview         Process 10 random photos only
#   --dry-run         Report without writing
#   --no-faces        Skip face detection (placeholder)
#   --min-cluster-size N  HDBSCAN param (default: 5)
#   --min-samples N       HDBSCAN param (default: 3)
# ─────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="vision-photo-organizer:latest"
DOCKERFILE="$SCRIPT_DIR/Dockerfile"

# ── Colors for output ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

die() {
    echo -e "${RED}ERROR: $*${NC}" >&2
    exit 1
}

info()  { echo -e "${GREEN}$*${NC}"; }
warn()  { echo -e "${YELLOW}$*${NC}"; }

# ── Parse arguments ──
# We pass unknown args through to the container
INPUT_DIR=""
OUTPUT_DIR=""
FORCE_REBUILD=0
DOCKER_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input|-i)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --force-rebuild)
            FORCE_REBUILD=1
            shift
            ;;
        --help|-h)
            head -20 "$0" | sed 's/^# //; s/^#$//; s/^#!/usage/'
            exit 0
            ;;
        *)
            DOCKER_ARGS+=("$1")
            shift
            ;;
    esac
done

# ── Validate ──
[[ -n "$INPUT_DIR" ]]  || die "--input is required"
[[ -n "$OUTPUT_DIR" ]] || die "--output is required"

INPUT_DIR="$(realpath "$INPUT_DIR")"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR" 2>/dev/null || echo "$OUTPUT_DIR")"

[[ -d "$INPUT_DIR" ]] || die "Input directory does not exist: $INPUT_DIR"
mkdir -p "$OUTPUT_DIR" || die "Cannot create output directory: $OUTPUT_DIR"

# ── GPU requirement (MANDATORY) ──
GPU_FLAGS=""
if command -v nvidia-smi &>/dev/null; then
    if docker info 2>/dev/null | grep -qi "nvidia"; then
        info "NVIDIA GPU detected — enabling GPU acceleration"
        GPU_FLAGS="--gpus all"
    else
        die "nvidia-smi found but nvidia-container-toolkit not detected in Docker." \
            "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi
else
    die "No NVIDIA GPU detected — GPU is REQUIRED. Install NVIDIA drivers and nvidia-container-toolkit."
fi

# ── Build image ──
if [[ $FORCE_REBUILD -eq 1 ]]; then
    info "Rebuilding Docker image ($IMAGE_NAME)..."
    docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" "$SCRIPT_DIR" \
        --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
        || die "Docker build failed"
elif ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    info "Building Docker image ($IMAGE_NAME)..."
    docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" "$SCRIPT_DIR" \
        --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
        || die "Docker build failed"
else
    info "Image $IMAGE_NAME already exists (use --force-rebuild to rebuild)"
fi

# ── Run container ──
info "Processing photos..."
info "  Input:  $INPUT_DIR"
info "  Output: $OUTPUT_DIR"
[[ ${#DOCKER_ARGS[@]} -gt 0 ]] && info "  Extra args: ${DOCKER_ARGS[*]}"

set -x
docker run --rm \
    $GPU_FLAGS \
    -v "$INPUT_DIR:/data/input:ro" \
    -v "$OUTPUT_DIR:/data/output" \
    "$IMAGE_NAME" \
    --input /data/input --output /data/output \
    "${DOCKER_ARGS[@]}"
set +x

# ── Summary ──
GALLERY="$OUTPUT_DIR/index.html"
if [[ -f "$GALLERY" ]]; then
    echo ""
    info "✅ Done! Gallery: file://$GALLERY"
else
    # Check if --dry-run was used — no gallery in that case
    if [[ " ${DOCKER_ARGS[*]} " == *" --dry-run "* ]] || [[ " ${DOCKER_ARGS[*]} " == *" --dry-run" ]]; then
        info "✅ Dry run complete. No files were written."
    else
        warn "Gallery not found at $GALLERY — check container output above for errors."
    fi
fi
