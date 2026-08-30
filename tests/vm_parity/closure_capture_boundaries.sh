#!/usr/bin/env bash
# Execute closure-capture boundaries on native, VM source, and VM ESKB paths.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
ESHKOL_RUN="$BUILD_DIR/eshkol-run"
VM="$BUILD_DIR/eshkol-vm-standalone-test"
WORK_DIR="${1:?usage: $0 <work-dir>}"
TRACE_DIR="${TRACE_DIR:-$ROOT_DIR/scripts/icc_traces}"
TRACE_FILE="$TRACE_DIR/closure_capture_boundaries.jsonl"
mkdir -p "$TRACE_DIR"
: > "$TRACE_FILE"

emit_trace() {
    printf '{"kind":"vm_parity","name":"closure_capture_%s","value":"PASS","snippet":"native/vm-src/vm-eskb boundary matched","confidence":0.95}\n' "$1" >> "$TRACE_FILE"
}

python3 "$ROOT_DIR/tests/vm_parity/generate_closure_capture_regressions.py" "$WORK_DIR"

for count in 255 256 257 4096 65536; do
    source_file="$WORK_DIR/closure_capture_${count}.esk"
    eskb_file="$WORK_DIR/closure_capture_${count}.eskb"
    expected=$((count - 1))

    native_out="$WORK_DIR/native_${count}.out"
    vm_out="$WORK_DIR/vm_${count}.out"
    eskb_out="$WORK_DIR/eskb_${count}.out"

    "$ESHKOL_RUN" --no-stdlib -r "$source_file" >"$native_out" 2>"$WORK_DIR/native_${count}.err"
    ESHKOL_VM_NO_DISASM=1 "$VM" "$source_file" >"$vm_out" 2>"$WORK_DIR/vm_${count}.err"
    "$ESHKOL_RUN" --profile hosted-vm --emit-eskb "$eskb_file" "$source_file" \
        >"$WORK_DIR/emit_${count}.out" 2>"$WORK_DIR/emit_${count}.err"
    test -s "$eskb_file"
    ESHKOL_VM_NO_DISASM=1 "$VM" "$eskb_file" >"$eskb_out" 2>"$WORK_DIR/eskb_${count}.err"

    for result in "$native_out" "$vm_out" "$eskb_out"; do
        test "$(grep -Ec "^\\(${expected}\\)$" "$result")" -eq 1
        test "$(grep -Ec "^${expected}$" "$result")" -eq 1
    done
    emit_trace "$count"
    echo "PASS: ${count} captures native/vm-src/vm-eskb (expected ${expected} twice)"
done

echo "PASS: closure capture boundaries 255, 256, 257, 4096, 65536"
