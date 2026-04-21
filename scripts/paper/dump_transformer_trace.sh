#!/usr/bin/env bash
# dump_transformer_trace.sh — run the compiled transformer (weight-matrix
# matmul path) on the 74-program verification suite and emit per-step
# state traces in the same JSONL schema as dump_vm_trace.sh.
#
# The transformer executes each program by one forward pass per VM step:
# state_{t+1} = transformer_forward(state_t, opcode_at_pc_t). Per-step
# state extraction reads the state vector's partitioned layout (PC, SP,
# TOS, SOS, registers, memory, tape, flags).
#
# Usage:
#   bash scripts/paper/dump_transformer_trace.sh [output_file]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT="${1:-artifacts/paper/outputs/transformer-traces.jsonl}"
mkdir -p "$(dirname "$OUTPUT")"

BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build-paper}"
PROGRAM_SUITE="${PROGRAM_SUITE:-$REPO_ROOT/tests/sdnc/programs}"
WEIGHTS="${WEIGHTS:-$REPO_ROOT/artifacts/paper/outputs/weights.qlmw}"

# TODO (SDNC paper artifact): the transformer's per-step state extraction
# entry point needs a standalone runner similar to the reference VM's
# test_vm. The qllm_interpreter.c has the forward path; wrap it to emit
# per-step JSONL matching the reference schema.

if [[ ! -d "$PROGRAM_SUITE" ]]; then
    echo "  NOTE: program suite not present (TODO — see dump_vm_trace.sh)."
    echo '{"status":"todo","message":"transformer runner not wired"}' > "$OUTPUT"
    exit 0
fi

if [[ ! -x "$BUILD_DIR/transformer_runner" ]]; then
    echo "  NOTE: transformer_runner binary not built (TODO)."
    echo '{"status":"todo","message":"transformer_runner target missing"}' > "$OUTPUT"
    exit 0
fi

echo '' > "$OUTPUT"
count=0
for program in "$PROGRAM_SUITE"/*.esk; do
    name="$(basename "$program" .esk)"
    "$BUILD_DIR/transformer_runner" --weights "$WEIGHTS" \
        --program "$program" --trace-jsonl >> "$OUTPUT" || {
        echo "  WARN: $name transformer trace failed"
    }
    count=$((count + 1))
done

echo "  wrote $count program transformer traces to $OUTPUT"
wc -l "$OUTPUT"
shasum -a 256 "$OUTPUT"
