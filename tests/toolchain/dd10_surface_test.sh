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

run_probe() {
    local name="$1"
    local source="$2"
    local expected="$3"
    shift 3
    local defines=("$@")
    local vm_defines=""
    local i=0
    while [ "$i" -lt "${#defines[@]}" ]; do
        if [ "${defines[$i]}" = "-D" ] &&
           [ "$((i + 1))" -lt "${#defines[@]}" ]; then
            [ -z "$vm_defines" ] || vm_defines="$vm_defines,"
            vm_defines="$vm_defines${defines[$((i + 1))]}"
            i=$((i + 2))
        else
            i=$((i + 1))
        fi
    done
    local actual

    actual="$($RUN -n "${defines[@]}" -r "$source" 2>&1)" \
        || fail "$name: native JIT rejected probe"
    actual="$(printf '%s\n' "$actual" | sed '/^$/d')"
    [ "$actual" = "$expected" ] || fail "$name: native JIT output: $actual"

    local native_bin="$ESHKOL_TEST_TMPDIR/$name-native"
    "$RUN" -n "${defines[@]}" "$source" -o "$native_bin" \
        >"$ESHKOL_TEST_TMPDIR/$name-aot-build.log" 2>&1 \
        || fail "$name: native AOT rejected probe"
    actual="$($native_bin 2>&1)" || fail "$name: native AOT executable failed"
    actual="$(printf '%s\n' "$actual" | sed '/^$/d')"
    [ "$actual" = "$expected" ] || fail "$name: native AOT output: $actual"

    actual="$(cd "$ROOT" && ESHKOL_COMMAND_DEFINES="$vm_defines" \
        "$VM_BIN" "$source" 2>&1)" \
        || fail "$name: VM source execution failed"
    actual="$(printf '%s\n' "$actual" | sed '/^$/d')"
    case "$actual" in
        *"$expected"*) ;;
        *) fail "$name: VM source output: $actual" ;;
    esac
}

run_probe "dd10-scope" "$ROOT/tests/toolchain/dd10_scope_probe.esk" \
    $'(7 hidden 99 22 99)'
run_probe "dd10-load" "$ROOT/tests/toolchain/dd10_load_probe.esk" \
    $'(loaded-private 11)'
run_probe "dd10-cond-expand" "$ROOT/tests/toolchain/dd10_cond_expand_probe.esk" \
    $'and=alpha\nor=beta' "-D" "ALPHA=1" "-D" "BETA=two"
run_probe "dd10-cond-expand-fallback" "$ROOT/tests/toolchain/dd10_cond_expand_probe.esk" \
    $'or=beta\nor=beta' "-D" "BETA=two"

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
