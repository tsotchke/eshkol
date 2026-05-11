#!/usr/bin/env bash
# dump_vm_trace.sh — emit per-step JSONL trace from the reference C VM
# for the SDNC paper's three-way verification suite.
#
# The programs are inline test() invocations in lib/backend/weight_matrices.c
# (the paper artifact itself). The same binary runs all three paths
# (reference C / simulated transformer / matrix forward) and emits per-step
# trace lines for the reference and matrix paths via --trace-vm /
# --trace-transformer flags.
#
# This script writes ONLY the reference-VM trace; --trace-transformer is
# routed to /dev/null. To produce both traces in a single run (faster), see
# run_paper_suite.sh which calls weight_matrices once with both flags.
#
# Usage:
#   bash scripts/paper/dump_vm_trace.sh [output_file]
#
# Output: JSONL at $1 (default: artifacts/paper/outputs/vm-traces.jsonl)
# Each record: {program, program_id, step, pc, sp, tos, sos, output, opcode,
#               is_native, registers, memory, tape, flags}

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT="${1:-artifacts/paper/outputs/vm-traces.jsonl}"
mkdir -p "$(dirname "$OUTPUT")"

BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build-paper}"

# Build weight_matrices. (Same target as export_weights.sh.)
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

echo "  dumping reference-VM trace → $OUTPUT"
"$BUILD_DIR/tools/weight_matrices" \
    --trace-vm "$OUTPUT" \
    --trace-transformer /dev/null \
    > "$BUILD_DIR.vm_trace.log" 2>&1 || {
        echo "  weight_matrices failed; tail of log:"
        tail -20 "$BUILD_DIR.vm_trace.log" | sed 's/^/    /'
        exit 1
    }

if ! grep -Eq "=== Results: [0-9]+ passed, 0 failed ===" "$BUILD_DIR.vm_trace.log"; then
    echo "  WARNING: did not see an all-pass verification line; trace may be incomplete."
    tail -10 "$BUILD_DIR.vm_trace.log" | sed 's/^/    /'
    exit 1
fi

echo "  $(wc -l < "$OUTPUT" | tr -d ' ') trace lines written"
shasum -a 256 "$OUTPUT" 2>/dev/null || sha256sum "$OUTPUT"
