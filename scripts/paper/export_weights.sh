#!/usr/bin/env bash
# export_weights.sh — regenerate the QLMW v3 weight matrices from the ISA specification.
#
# The construction is deterministic: same ISA in, bit-identical weights out on any
# IEEE 754 float32 platform. No training, no random seeds.
#
# Usage:
#   bash scripts/paper/export_weights.sh [output_file]
#
# Output: QLMW v3 binary at $1
# Default: artifacts/paper/outputs/weights.qlmw

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT="${1:-artifacts/paper/outputs/weights.qlmw}"
mkdir -p "$(dirname "$OUTPUT")"

# Use a dedicated build dir for the paper artifact so the user's normal
# build tree is unaffected and so the run is hermetic.
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build-paper}"
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "  building weight_matrices (paper config)..."
    cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
        > "$BUILD_DIR.cmake.log" 2>&1 || {
            echo "  cmake configuration failed; see $BUILD_DIR.cmake.log"
            exit 1
        }
fi
cmake --build "$BUILD_DIR" --target weight_matrices -j \
    > "$BUILD_DIR.build.log" 2>&1 || {
        echo "  build failed; see $BUILD_DIR.build.log"
        exit 1
    }

# Run the weight matrix construction. The binary self-tests via the
# three-way verification suite (reference C interpreter / simulated transformer
# / actual matrix forward pass through W @ x + b) and writes the QLMW v3 binary
# only on zero failures. ESHKOL_WEIGHTS_OUT redirects the output path.
echo "  running verification + weight export..."
ESHKOL_WEIGHTS_OUT="$OUTPUT" "$BUILD_DIR/tools/weight_matrices" \
    > "$BUILD_DIR.weights.log" 2>&1 || {
        echo "  verification or export failed; see $BUILD_DIR.weights.log"
        echo "  tail of log:"
        tail -20 "$BUILD_DIR.weights.log" | sed 's/^/    /'
        exit 1
    }

# Confirm the all-pass line is present — paper claim §4.4.
result_line="$(grep -E "=== Results: [0-9]+ passed, 0 failed ===" "$BUILD_DIR.weights.log" | tail -1 || true)"
if [[ -z "$result_line" ]]; then
    echo "  WARNING: did not see an all-pass verification line."
    echo "  tail of log:"
    tail -10 "$BUILD_DIR.weights.log" | sed 's/^/    /'
    exit 1
fi
passed="$(printf '%s\n' "$result_line" | sed -E 's/.*Results: ([0-9]+) passed, 0 failed.*/\1/')"

echo "  $passed/$passed verification passes (reference == simulated == matrix forward)."

if [[ ! -f "$OUTPUT" ]]; then
    echo "  ERROR: expected $OUTPUT to exist after run."
    exit 1
fi

bytes=$(wc -c < "$OUTPUT" | tr -d ' ')
echo "  weights at: $OUTPUT ($bytes bytes)"

# Print the SHA-256 (paper §4.5 expected-checksums table is filled by this).
if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$OUTPUT"
elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$OUTPUT"
fi
