#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────
# Photo Organizer — Docker wrapper
# ─────────────────────────────────────────────────────────────
# Usage:
#   ./organize.sh --input /path/to/photos --output /path/to/out [OPTIONS]
#
# Options:
#   --input, -i       Input directory or smb:// URI (required)
#   --output, -o      Output directory (required)
#   --preview         Process 10 random photos only
#   --dry-run         Report without writing
#   --no-gallery      Skip HTML gallery generation
#   --no-recursive-search  Only scan root of input directory
#   --output-mode [flat|folders]  Output structure (default: flat)
#   --no-faces        Skip face detection (placeholder)
#   --model           Vision model: dinov2-large (default), dinov2-base, clip, siglip, hybrid
#   --umap-components N  UMAP target dimensions (default: 20)
#   --min-cluster-size N  HDBSCAN minimum cluster size (default: 15)
#
# SMB/CIFS support:
#   Use --input "smb://server/share/path" to mount a network share.
#   Set SMB_USER and SMB_PASS env vars for authenticated shares.
#   Port is extracted from URI (e.g. smb://server:443/share).
#   Requires cifs-utils on the host: sudo apt install cifs-utils
# ─────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="vision-photo-organizer:latest"
DOCKERFILE="$SCRIPT_DIR/Dockerfile"

# Cargar variables de entorno desde .env si existe
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

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

# ── SMB/CIFS mount handling (Docker Volumes) ──
SMB_VOLUMES=()
DOCKER_MOUNTS=()
CONTAINER_INPUT_DIR="/data/input"
CONTAINER_OUTPUT_DIR="/data/output"

cleanup_smb_mounts() {
    local vol
    for vol in "${SMB_VOLUMES[@]:-}"; do
        [[ -z "$vol" ]] && continue
        info "Removing SMB Docker volume: $vol"
        docker volume rm "$vol" >/dev/null 2>&1 || true
    done
}
trap cleanup_smb_mounts EXIT

# Shared SMB mounter: _mount_smb <label> <uri> <out_subpath_var> <out_vol_var>
_mount_smb() {
    local label="$1" uri="$2" out_subpath_var="$3" out_vol_var="$4"
    local unc_path decoded_path smb_port port_opts smb_opts
    local share subpath vol_name

    unc_path="//${uri#smb://}"
    decoded_path=$(python3 -c "import sys, urllib.parse; print(urllib.parse.unquote(sys.argv[1]))" "$unc_path" 2>/dev/null || echo "$unc_path")

    if [[ "$decoded_path" =~ ^(//[^/]+/[^/]+)(/.*)?$ ]]; then
        share="${BASH_REMATCH[1]}"
        subpath="${BASH_REMATCH[2]:1}" # remove leading slash
    else
        die "Invalid SMB URI format (must include server and share): $decoded_path"
    fi

    smb_port=""; port_opts=""
    if [[ "$share" =~ ^//([^:]+):([0-9]+)(/.*) ]]; then
        local server="${BASH_REMATCH[1]}"
        smb_port="${BASH_REMATCH[2]}"
        local share_name="${BASH_REMATCH[3]}"
        share="//${server}${share_name}"
        port_opts=",port=$smb_port"
    fi

    if [[ -n "${SMB_USER:-}" ]]; then
        smb_opts="username=$SMB_USER,password=${SMB_PASS:-},uid=$(id -u),gid=$(id -g)$port_opts"
        [[ -z "${SMB_PASS:-}" ]] && warn "SMB_PASS is empty — some shares allow this, some don't."
    else
        smb_opts="guest,uid=$(id -u),gid=$(id -g)$port_opts"
    fi

    vol_name="photo_smb_${label}_$$_${RANDOM}"
    info "Creating Docker volume for SMB share: $share"
    
    if ! docker volume create --driver local --opt type=cifs --opt device="$share" --opt o="$smb_opts" "$vol_name" >/dev/null; then
        die "Failed to create Docker volume for $share (Check credentials and network)"
    fi

    SMB_VOLUMES+=("$vol_name")

    if [[ "$label" == "output" && -n "$subpath" ]]; then
        info "Ensuring output subdirectory exists..."
        # Use a lightweight container to create the directory inside the share
        docker run --rm -v "$vol_name:/mnt" alpine mkdir -p "/mnt/$subpath" 2>/dev/null || true
    fi

    printf -v "$out_subpath_var" "%s" "$subpath"
    printf -v "$out_vol_var" "%s" "$vol_name"
}

if [[ "$INPUT_DIR" == smb://* ]]; then
    _mount_smb "input" "$INPUT_DIR" "SMB_IN_SUB" "SMB_IN_VOL"
    DOCKER_MOUNTS+=("-v" "$SMB_IN_VOL:/data/input_share")
    CONTAINER_INPUT_DIR="/data/input_share/$SMB_IN_SUB"
else
    INPUT_DIR="$(realpath "$INPUT_DIR")"
    [[ -d "$INPUT_DIR" ]] || die "Input directory does not exist: $INPUT_DIR"
    DOCKER_MOUNTS+=("-v" "$INPUT_DIR:/data/input")
    CONTAINER_INPUT_DIR="/data/input"
fi

if [[ "$OUTPUT_DIR" == smb://* ]]; then
    _mount_smb "output" "$OUTPUT_DIR" "SMB_OUT_SUB" "SMB_OUT_VOL"
    DOCKER_MOUNTS+=("-v" "$SMB_OUT_VOL:/data/output_share")
    CONTAINER_OUTPUT_DIR="/data/output_share/$SMB_OUT_SUB"
else
    OUTPUT_DIR="$(realpath "$OUTPUT_DIR" 2>/dev/null || echo "$OUTPUT_DIR")"
    mkdir -p "$OUTPUT_DIR" || die "Cannot create output directory: $OUTPUT_DIR"
    DOCKER_MOUNTS+=("-v" "$OUTPUT_DIR:/data/output")
    CONTAINER_OUTPUT_DIR="/data/output"
fi

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
    info "Image $IMAGE_NAME already exists — verifying GPU support..."
    GPU_OUTPUT=$(docker run --rm --gpus all --entrypoint python "$IMAGE_NAME" \
        -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>&1)
    GPU_EXIT=$?
    
    if [ "$GPU_EXIT" -eq 0 ]; then
        info "GPU support confirmed in existing image"
    else
        warn "Existing image does NOT have working GPU support."
        if [ -z "$GPU_OUTPUT" ]; then
            # Exit 1 with no output → PyTorch cleanly reports no CUDA device
            warn "PyTorch reports no CUDA device available in the container."
            warn "The image was likely built without GPU support."
            warn "Re-run with --force-rebuild:  ./organize.sh --force-rebuild ..."
        elif echo "$GPU_OUTPUT" | grep -qi "could not select device driver\|Error response from daemon"; then
            warn "Docker runtime error detected (not necessarily the image):"
            warn "$GPU_OUTPUT"
            warn "Check that nvidia-container-toolkit is installed and Docker is configured for GPU access."
        else
            warn "Unexpected error during GPU verification:"
            warn "$GPU_OUTPUT"
            warn "Try --force-rebuild or check Docker/nvidia setup."
        fi
        die "GPU-required tool cannot proceed without GPU access."
    fi
fi

# ── Run container ──
info "Processing photos..."
info "  Input:  $INPUT_DIR"
info "  Output: $OUTPUT_DIR"
[[ ${#DOCKER_ARGS[@]} -gt 0 ]] && info "  Extra args: ${DOCKER_ARGS[*]}"

set -x
docker run --rm \
    $GPU_FLAGS \
    -e "HOST_UID=$(id -u)" \
    -e "HOST_GID=$(id -g)" \
    "${DOCKER_MOUNTS[@]}" \
    "$IMAGE_NAME" \
    --input "$CONTAINER_INPUT_DIR" --output "$CONTAINER_OUTPUT_DIR" \
    "${DOCKER_ARGS[@]}"
set +x

# ── Summary ──
GALLERY="$OUTPUT_DIR/index.html"
if [[ -f "$GALLERY" ]]; then
    echo ""
    info "✅ Done! Gallery: file://$GALLERY"
else
    # Check if --dry-run or --no-gallery was used — no gallery in those cases
    if [[ " ${DOCKER_ARGS[*]} " == *" --dry-run "* ]] || [[ " ${DOCKER_ARGS[*]} " == *" --dry-run" ]]; then
        info "✅ Dry run complete. No files were written."
    elif [[ " ${DOCKER_ARGS[*]} " == *" --no-gallery "* ]] || [[ " ${DOCKER_ARGS[*]} " == *" --no-gallery" ]]; then
        info "✅ Done! (gallery skipped)"
    else
        warn "Gallery not found at $GALLERY — check container output above for errors."
    fi
fi
