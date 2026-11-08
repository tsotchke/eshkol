#!/usr/bin/env bash
# Run one ipc-unix-pending .esk acceptance case under JIT or AOT. Never uses
# /tmp: the scratch directory is a build-tree path passed in by CMake.
set -euo pipefail
if [ "$#" -ne 5 ]; then
    echo "usage: $0 <eshkol-run> <jit|aot> <scenario> <scratch-dir> <esk-file>" >&2
    exit 2
fi
RUNNER="$1"
MODE="$2"
SCENARIO="$3"
SCRATCH="$4"
ESK="$5"
mkdir -p "$SCRATCH"
export ESHKOL_UNIX_SOCKET_SCRATCH="$SCRATCH"
export ESHKOL_UNIX_SOCKET_SCENARIO="$SCENARIO"
export ESHKOL_JIT_CACHE=0
case "$MODE" in
    jit)
        exec "$RUNNER" -r "$ESK"
        ;;
    aot)
        exe="$SCRATCH/${SCENARIO}-aot"
        "$RUNNER" -O0 -o "$exe" "$ESK"
        exec "$exe"
        ;;
    *)
        echo "unknown mode: $MODE" >&2
        exit 2
        ;;
esac
