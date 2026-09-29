#!/usr/bin/env bash
set -eu

if [ "$#" -ne 3 ]; then
    echo "usage: $0 <eshkol-run> <eshkol-vm-standalone-test> <build-dir>" >&2
    exit 2
fi

ESH="$1"
VM="$2"
BUILD_DIR="$3"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
TEST_DIR="$ROOT_DIR/tests/bindings"
WORK_DIR="$(mktemp -d "$BUILD_DIR/nested-shorthand-cells.XXXXXX")"
cleanup() {
    rc=$?
    if [ "$rc" -eq 0 ]; then
        rm -rf "$WORK_DIR"
    else
        echo "FAIL: captured lane logs in $WORK_DIR" >&2
    fi
}
trap cleanup EXIT

compare_case() {
    name="$1"
    source="$2"
    expected="$3"
    printf '%s\n' "$expected" > "$WORK_DIR/expected.out"

    ESHKOL_JIT_CACHE=0 "$ESH" -r "$source" -L"$BUILD_DIR" >"$WORK_DIR/native-jit.out" 2>"$WORK_DIR/native-jit.err"
    "$ESH" "$source" -o "$WORK_DIR/native-aot" -L"$BUILD_DIR" >"$WORK_DIR/native-aot-build.out" 2>"$WORK_DIR/native-aot-build.err"
    "$WORK_DIR/native-aot" >"$WORK_DIR/native-aot.out" 2>"$WORK_DIR/native-aot.err"
    ESHKOL_VM_NO_DISASM=1 "$VM" "$source" >"$WORK_DIR/vm-src.out" 2>"$WORK_DIR/vm-src.err"
    "$ESH" --profile hosted-vm --emit-eskb "$WORK_DIR/$name.eskb" "$source" -L"$BUILD_DIR" >"$WORK_DIR/eskb-build.out" 2>"$WORK_DIR/eskb-build.err"
    ESHKOL_VM_NO_DISASM=1 "$VM" "$WORK_DIR/$name.eskb" >"$WORK_DIR/vm-eskb.out" 2>"$WORK_DIR/vm-eskb.err"

    for lane in native-jit native-aot vm-src vm-eskb; do
        if ! diff -u "$WORK_DIR/expected.out" "$WORK_DIR/$lane.out"; then
            echo "FAIL: $name $lane output differed" >&2
            exit 1
        fi
    done
}

compare_case nested_shorthand_function_setbang \
    "$TEST_DIR/nested_shorthand_function_setbang_test.esk" 42
compare_case nested_shorthand_function_cells \
    "$TEST_DIR/nested_shorthand_function_cells_test.esk" $'12\n23\n1\n42\n42\n13'

# Undefined assignment must fail compilation and must not leave bytecode.
rm -f "$WORK_DIR/undefined.eskb"
set +e
"$ESH" --profile hosted-vm --emit-eskb "$WORK_DIR/undefined.eskb" \
    "$TEST_DIR/shorthand_function_setbang_undefined_negative.esk" \
    -L"$BUILD_DIR" >"$WORK_DIR/undefined.out" 2>"$WORK_DIR/undefined.err"
undefined_rc=$?
set -e
if [ "$undefined_rc" -eq 0 ] || [ -e "$WORK_DIR/undefined.eskb" ] || \
   ! grep -q "ERROR: set! on undefined variable" "$WORK_DIR/undefined.err"; then
    echo "FAIL: undefined set! did not fail closed (rc=$undefined_rc)" >&2
    cat "$WORK_DIR/undefined.err" >&2
    exit 1
fi

echo "PASS: nested shorthand closures preserve native/VM parity and reject undefined set!"
