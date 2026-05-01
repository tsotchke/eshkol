#!/usr/bin/env bash
# dump_transformer_trace.sh — emit per-step JSONL trace from the matrix
# forward pass (W @ x + b) on the SDNC paper's 74-program suite.
#
# Mirrors dump_vm_trace.sh but routes the reference-VM output to /dev/null
# and captures the matrix path via --trace-transformer.
#
# Usage:
#   bash scripts/paper/dump_transformer_trace.sh [output_file]
#
# Output: JSONL at $1 (default: artifacts/paper/outputs/transformer-traces.jsonl)
# Schema matches dump_vm_trace.sh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT="${1:-artifacts/paper/outputs/transformer-traces.jsonl}"
mkdir -p "$(dirname "$OUTPUT")"

BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build-paper}"

if [[ ! -x "$BUILD_DIR/tools/weight_matrices" ]]; then
    echo "  building weight_matrices (paper config)..."
    cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
        > "$BUILD_DIR.cmake.log" 2>&1 || {
            echo "  cmake configuration failed; see $BUILD_DIR.cmake.log"
            exit 1
        }
    cmake --build "$BUILD_DIR" --target weight_matrices -j \
        > "$BUILD_DIR.build.log" 2>&1 || {
            echo "  build failed; see $BUILD_DIR.build.log"
            exit 1
        }
fi

echo "  dumping matrix-forward trace → $OUTPUT"
"$BUILD_DIR/tools/weight_matrices" \
    --trace-vm /dev/null \
    --trace-transformer "$OUTPUT" \
    > "$BUILD_DIR.tf_trace.log" 2>&1 || {
        echo "  weight_matrices failed; tail of log:"
        tail -20 "$BUILD_DIR.tf_trace.log" | sed 's/^/    /'
        exit 1
    }

if ! grep -q "74 passed, 0 failed" "$BUILD_DIR.tf_trace.log"; then
    echo "  WARNING: did not see '74 passed, 0 failed'; trace may be incomplete."
    tail -10 "$BUILD_DIR.tf_trace.log" | sed 's/^/    /'
    exit 1
fi

echo "  $(wc -l < "$OUTPUT" | tr -d ' ') trace lines written"
shasum -a 256 "$OUTPUT" 2>/dev/null || sha256sum "$OUTPUT"
