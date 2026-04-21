#!/usr/bin/env bash
# dump_vm_trace.sh — run the reference C bytecode VM on the 74-program
# verification suite and emit per-step state traces as JSONL.
#
# Each JSONL record captures one VM step:
#   {
#     "program": "<name>",
#     "step": <int>,
#     "pc": <int>, "sp": <int>, "tos": <float>, "sos": <float>,
#     "opcode": <int>, "is_native": <bool>,
#     "registers": [<float>, ...],
#     "memory":    [<float>, ...],
#     "tape":      [<float>, ...],
#     "flags":     {"zero": <bool>, ...}
#   }
#
# Usage:
#   bash scripts/paper/dump_vm_trace.sh [output_file]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT="${1:-artifacts/paper/outputs/vm-traces.jsonl}"
mkdir -p "$(dirname "$OUTPUT")"

BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build-paper}"

# Standalone VM build path per CONTRIBUTING.md
if [[ ! -x "$BUILD_DIR/test_vm" ]]; then
    echo "  compiling standalone reference VM..."
    mkdir -p "$BUILD_DIR"
    gcc -O2 -std=c11 -w -DESHKOL_VM_TRACE_JSONL=1 \
        lib/backend/eshkol_vm.c -o "$BUILD_DIR/test_vm" -lm -lpthread
fi

# TODO (SDNC paper artifact): the 74-program verification suite needs a
# canonical list of programs + inputs. For the v1.1.13 release this lives
# inside lib/backend/weight_matrices.c's main() or equivalent. Wire that
# list into a standalone runner that emits JSONL per step.
#
# Current status: the reference VM binary exists and can run programs;
# the per-step JSONL emitter needs to be added via the
# ESHKOL_VM_TRACE_JSONL flag at compile time, and each of the 74 test
# programs needs a per-program invocation here.

PROGRAM_SUITE="${PROGRAM_SUITE:-$REPO_ROOT/tests/sdnc/programs}"

if [[ ! -d "$PROGRAM_SUITE" ]]; then
    echo "  NOTE: program suite directory $PROGRAM_SUITE not present (TODO)."
    echo "  Expected: 74 .esk programs used in three-way verification."
    echo "  Emitting marker at $OUTPUT"
    echo '{"status":"todo","message":"program suite not wired"}' > "$OUTPUT"
    exit 0
fi

echo '' > "$OUTPUT"
count=0
for program in "$PROGRAM_SUITE"/*.esk; do
    name="$(basename "$program" .esk)"
    ESHKOL_VM_NO_DISASM=1 "$BUILD_DIR/test_vm" --trace-jsonl "$program" >> "$OUTPUT" || {
        echo "  WARN: $name trace failed"
    }
    count=$((count + 1))
done

echo "  wrote $count program traces to $OUTPUT"
wc -l "$OUTPUT"
shasum -a 256 "$OUTPUT"
