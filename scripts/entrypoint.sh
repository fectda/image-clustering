#!/usr/bin/env bash
set -euo pipefail

# Entrypoint wrapper that creates a user matching the host UID/GID
# so output files are owned by the host user, not root.
#
# Pass HOST_UID and HOST_GID as env vars, e.g.:
#   docker run -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) ...

# GPU validation — verify CUDA is available before launching
check_gpu() {
    python -c "
import torch
if not torch.cuda.is_available():
    print('=' * 70, flush=True)
    print('ERROR: CUDA is not available — GPU is REQUIRED.', flush=True)
    print('=' * 70, flush=True)
    print('Possible causes:', flush=True)
    print('  1. Image was built without CUDA support', flush=True)
    print('     → Rebuild: docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 .', flush=True)
    print('  2. Container running without GPU access', flush=True)
    print('     → Add --gpus all to docker run, or use ./organize.sh', flush=True)
    print('  3. NVIDIA Container Toolkit not installed', flush=True)
    print('     → Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/', flush=True)
    print('  4. No NVIDIA GPU on this machine', flush=True)
    print('     → This tool requires an NVIDIA GPU with CUDA support.', flush=True)
    print('=' * 70, flush=True)
    exit(1)
"
}

HOST_UID="${HOST_UID:-}"
HOST_GID="${HOST_GID:-}"

if [[ -n "$HOST_UID" && -n "$HOST_GID" ]]; then
    # Create a user matching the host UID/GID
    groupadd -g "$HOST_GID" hostgroup 2>/dev/null || true
    useradd -u "$HOST_UID" -g "$HOST_GID" -m -s /bin/bash hostuser 2>/dev/null || true

    # Torch inductor calls getpass.getuser() which needs the user in /etc/passwd
    export TORCHINDUCTOR_CACHE_DIR="/tmp/torch-cache-${HOST_UID}"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

    # Run as the host user (args are always subcommand args for photo_organizer.main)
    check_gpu
    exec gosu hostuser python -m photo_organizer.main "$@"
fi

# No HOST_UID — run as root (default behavior)
check_gpu
exec python -m photo_organizer.main "$@"
