#!/usr/bin/env bash
# Verify that a continuation captured at the cumulative region-pin boundary
# is rejected on native and VM, without changing the existing diagnostics.
set -u
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
BUILD_DIR=${BUILD_DIR:-build}
RUN=${ESHKOL_RUN:-$ROOT/$BUILD_DIR/eshkol-run}
VM=${ESHKOL_VM:-$ROOT/$BUILD_DIR/eshkol-vm-standalone-test}
NATIVE_SRC=$ROOT/tests/memory/region_pin_budget_native_boundary.esk
VM_SRC=$ROOT/tests/memory/region_pin_budget_vm_boundary.esk
WORK_ROOT=${ESHKOL_SCRATCH_ROOT:-$ROOT/.scratch}
mkdir -p "$WORK_ROOT"
WORK=$(mktemp -d "$WORK_ROOT/eshkol-pin-budget.XXXXXX")
trap 'rm -rf "$WORK"' EXIT INT TERM

if [ ! -x "$RUN" ] || [ ! -x "$VM" ]; then
    echo "continuation_pin_budget_boundary_test.sh: required native/VM binaries missing" >&2
    exit 2
fi

PASS=0
FAIL=0
check() {
    if [ "$3" -eq 0 ]; then
        echo "PASSED tests/memory/continuation_pin_budget_boundary_test.sh::$1"
        PASS=$((PASS + 1))
    else
        echo "FAILED tests/memory/continuation_pin_budget_boundary_test.sh::$1 — $2"
        FAIL=$((FAIL + 1))
    fi
}

NATIVE_ERR=$WORK/native.err
set +e
"$RUN" -r "$NATIVE_SRC" >"$WORK/native.out" 2>"$NATIVE_ERR"
native_rc=$?
set -e
if [ "$native_rc" -ne 0 ] && grep -qF "continuation region-pin budget exceeded" "$NATIVE_ERR"; then rc=0; else rc=1; fi
check native_jit_boundary "native JIT accepted a continuation beyond the pin budget (rc=$native_rc)" "$rc"

NATIVE_EXE=$WORK/native-aot
set +e
"$RUN" -o "$NATIVE_EXE" "$NATIVE_SRC" >"$WORK/native-compile.out" 2>"$WORK/native-compile.err"
compile_rc=$?
if [ "$compile_rc" -eq 0 ] && [ -x "$NATIVE_EXE" ]; then
    "$NATIVE_EXE" >"$WORK/native-aot.out" 2>"$WORK/native-aot.err"
    native_aot_rc=$?
else
    native_aot_rc=127
fi
set -e
if [ "$native_aot_rc" -ne 0 ] && grep -qF "continuation region-pin budget exceeded" "$WORK/native-aot.err"; then rc=0; else rc=1; fi
check native_aot_boundary "native AOT accepted a continuation beyond the pin budget (rc=$native_aot_rc)" "$rc"

VM_ERR=$WORK/vm.err
set +e
ESHKOL_VM_NO_DISASM=1 ESHKOL_VM_REGION_QUIET=1 "$VM" "$VM_SRC" >"$WORK/vm.out" 2>"$VM_ERR"
vm_rc=$?
set -e
if [ "$vm_rc" -ne 0 ] && grep -qF "continuation region-pin budget exceeded" "$VM_ERR"; then rc=0; else rc=1; fi
check vm_boundary "VM accepted a continuation beyond the pin budget (rc=$vm_rc)" "$rc"

echo "continuation-pin-budget-boundary: $PASS passed, $FAIL failed"
if [ "$FAIL" -eq 0 ]; then
    echo "continuation_pin_budget_boundary_test.sh: PASS"
    exit 0
fi
echo "continuation_pin_budget_boundary_test.sh: FAIL"
exit 1
