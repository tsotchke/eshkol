#!/usr/bin/env bash
# Regression for outlined top-level AOT initialization. The same generated
# source is exercised through native JIT, native AOT, and the hosted VM.
set -u

BUILD_DIR="${1:?usage: aot_init_outline_parity_test.sh <build-dir> <work-root>}"
WORK_ROOT="${2:?usage: aot_init_outline_parity_test.sh <build-dir> <work-root>}"
ESHKOL_RUN="$BUILD_DIR/eshkol-run"
ESHKOL_VM="$BUILD_DIR/eshkol-vm-standalone-test"
SRC="$WORK_ROOT/aot-init-outline.esk"
AOT="$WORK_ROOT/aot-init-outline.bin"
ESKB="$WORK_ROOT/aot-init-outline.eskb"

[ -x "$ESHKOL_RUN" ] || { echo "missing eshkol-run: $ESHKOL_RUN" >&2; exit 1; }
[ -x "$ESHKOL_VM" ] || { echo "missing hosted VM: $ESHKOL_VM" >&2; exit 1; }
mkdir -p "$WORK_ROOT"
python3 "$BUILD_DIR/../bench/generate_large_single_file.py" \
    --defines 80 --out "$SRC"

JIT_OUTPUT="$($ESHKOL_RUN -r "$SRC" 2>"$WORK_ROOT/jit.err")" || {
    echo "native JIT failed" >&2; cat "$WORK_ROOT/jit.err" >&2; exit 1;
}
"$ESHKOL_RUN" -O 0 "$SRC" -o "$AOT" >"$WORK_ROOT/aot.log" 2>&1 || {
    echo "native AOT failed" >&2; cat "$WORK_ROOT/aot.log" >&2; exit 1;
}
AOT_OUTPUT="$("$AOT")" || { echo "native AOT executable failed" >&2; exit 1; }

"$ESHKOL_RUN" --profile hosted-vm --emit-eskb "$ESKB" "$SRC" \
    >"$WORK_ROOT/vm-emit.log" 2>&1 || {
    echo "hosted VM emission failed" >&2; cat "$WORK_ROOT/vm-emit.log" >&2; exit 1;
}
VM_OUTPUT_RAW="$("$ESHKOL_VM" "$ESKB")" || {
    echo "hosted VM execution failed" >&2; exit 1;
}
VM_OUTPUT="$(printf '%s\n' "$VM_OUTPUT_RAW" | sed -nE 's/^(-?[0-9]+(\.[0-9]+)?)$/\1/p' | tail -n 1)"
if [ -z "$VM_OUTPUT" ]; then
    echo "hosted VM produced no scalar result" >&2
    exit 1
fi

if [ "$JIT_OUTPUT" != "$AOT_OUTPUT" ] || [ "$JIT_OUTPUT" != "$VM_OUTPUT" ]; then
    echo "native JIT/AOT/VM output mismatch" >&2
    printf 'jit=%s\naot=%s\nvm=%s\n' "$JIT_OUTPUT" "$AOT_OUTPUT" "$VM_OUTPUT" >&2
    exit 1
fi

echo "PASS: outlined AOT initialization preserves native JIT/AOT/VM parity"
