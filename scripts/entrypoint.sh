#!/usr/bin/env bash
set -euo pipefail

# Entrypoint wrapper that creates a user matching the host UID/GID
# so output files are owned by the host user, not root.
#
# Pass HOST_UID and HOST_GID as env vars, e.g.:
#   docker run -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) ...

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
    exec gosu hostuser python -m photo_organizer.main "$@"
fi

# No HOST_UID — run as root (default behavior)
exec python -m photo_organizer.main "$@"
