#!/usr/bin/env bash
set -u

RUN="${1:-}"
BUILD_DIR="${2:-}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TEST_NAME="dd10_surface_test"

fail() {
    echo "FAIL: $TEST_NAME: $*" >&2
    exit 1
}

[ -x "$RUN" ] || fail "eshkol-run is not executable: $RUN"
[ -d "$BUILD_DIR" ] || fail "build directory not found: $BUILD_DIR"

export ESHKOL_TEST_TMP_ROOT="$ROOT/.scratch"
ISO="$ROOT/scripts/lib/test_isolation.sh"
[ -r "$ISO" ] || fail "test isolation helper is unavailable"
source "$ISO"
eshkol_test_isolation_init "$TEST_NAME"
trap eshkol_test_isolation_cleanup EXIT

SRC="$ROOT/tests/toolchain/dd10_feature_probe.esk"
NATIVE_JIT="$ESHKOL_TEST_TMPDIR/native-jit.out"
NATIVE_BIN="$ESHKOL_TEST_TMPDIR/native.bin"
VM_BC="$ESHKOL_TEST_TMPDIR/probe.eskb"
PIC_OBJ="$ESHKOL_TEST_TMPDIR/probe.o"
PRIV_SRC="$ROOT/tests/modules/visibility_fail_test.esk"
PUBLIC_SRC="$ROOT/tests/modules/visibility_test.esk"
PRIV_VM_BC="$ESHKOL_TEST_TMPDIR/private.eskb"
PUBLIC_VM_BC="$ESHKOL_TEST_TMPDIR/public.eskb"

actual="$($RUN -n -D DD10_FEATURE=1 -r "$SRC" 2>&1)" || fail "native JIT rejected -D"
[ "$actual" = "PASS: -D feature" ] || fail "native JIT output: $actual"

actual="$($RUN -n -r "$PUBLIC_SRC" 2>&1)" || fail "native JIT rejected exported module"
case "$actual" in
    *"exported double-it(5)"*"exported square(-3)"*) ;;
    *) fail "native JIT visibility output: $actual" ;;
esac

"$RUN" -n -D DD10_FEATURE=1 "$SRC" -o "$NATIVE_BIN" >"$ESHKOL_TEST_TMPDIR/native-build.log" 2>&1 \
    || fail "native AOT rejected -D"
[ -x "$NATIVE_BIN" ] || fail "native AOT did not produce an executable"
actual="$($NATIVE_BIN 2>&1)" || fail "native AOT executable failed"
[ "$actual" = "PASS: -D feature" ] || fail "native AOT output: $actual"

PUBLIC_NATIVE_BIN="$ESHKOL_TEST_TMPDIR/public-native"
"$RUN" -n "$PUBLIC_SRC" -o "$PUBLIC_NATIVE_BIN" \
    >"$ESHKOL_TEST_TMPDIR/public-native-build.log" 2>&1 \
    || fail "native AOT rejected exported module"
actual="$($PUBLIC_NATIVE_BIN 2>&1)" || fail "native AOT exported module failed"
case "$actual" in
    *"exported double-it(5)"*"exported square(-3)"*) ;;
    *) fail "native AOT visibility output: $actual" ;;
esac

"$RUN" -n -D DD10_FEATURE=1 --profile hosted-vm --emit-eskb "$VM_BC" "$SRC" \
    >"$ESHKOL_TEST_TMPDIR/vm-build.log" 2>&1 \
    || fail "VM ESKB compilation rejected -D"
VM_BIN="$BUILD_DIR/eshkol-vm-standalone-test"
[ -x "$VM_BIN" ] || VM_BIN="$BUILD_DIR/eshkol-vm-standalone"
[ -x "$VM_BIN" ] || fail "VM standalone executable is unavailable"
actual="$($VM_BIN "$VM_BC" 2>&1)" || fail "VM ESKB executable failed"
case "$actual" in
    *"PASS: -D feature"*) ;;
    *) fail "VM ESKB output: $actual" ;;
esac

"$RUN" -n --profile hosted-vm --emit-eskb "$PUBLIC_VM_BC" "$PUBLIC_SRC" \
    >"$ESHKOL_TEST_TMPDIR/public-vm-build.log" 2>&1 \
    || fail "VM rejected exported module"
actual="$($VM_BIN "$PUBLIC_VM_BC" 2>&1)" || fail "VM exported module failed"
case "$actual" in
    *"exported double-it(5)"*"exported square(-3)"*) ;;
    *) fail "VM visibility output: $actual" ;;
esac

"$RUN" -n -D DD10_FEATURE=1 -fPIC -c "$SRC" -o "$PIC_OBJ" \
    >"$ESHKOL_TEST_TMPDIR/pic-build.log" 2>&1 \
    || fail "AOT -fPIC object emission failed"
[ -f "$PIC_OBJ" ] || fail "AOT -fPIC did not produce an object"

if "$RUN" -n -r "$PRIV_SRC" >"$ESHKOL_TEST_TMPDIR/private-jit.log" 2>&1; then
    fail "native JIT accepted a private module binding"
fi

if "$RUN" -n "$PRIV_SRC" -o "$ESHKOL_TEST_TMPDIR/private-native" \
        >"$ESHKOL_TEST_TMPDIR/private-aot.log" 2>&1; then
    fail "native AOT accepted a private module binding"
fi

if (cd "$ROOT" && "$RUN" -n --profile hosted-vm \
        --emit-eskb "$PRIV_VM_BC" "$PRIV_SRC" \
        >"$ESHKOL_TEST_TMPDIR/private-vm.log" 2>&1); then
    fail "VM accepted a private module binding"
fi

echo "PASS: $TEST_NAME"
