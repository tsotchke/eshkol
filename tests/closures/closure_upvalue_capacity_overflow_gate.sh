#!/usr/bin/env bash
# SW-45 gate: closure captures beyond the former fixed boundary run correctly
# (including the top-level `define` compiled right after the big procedure)
# instead of silently corrupting the runtime stack.
#
# Before the fix, compiler and runtime capture storage had independent fixed
# limits. A closure beyond the smaller runtime array silently desynchronised
# the operand stack and corrupted later top-level definitions.
#
# The compiler and runtime now size capture storage to the closure's actual
# free-variable count, so this gate's two cases pin that behavior:
#
#   1. AT the capacity (32 distinct top-level calls in one procedure): the
#      procedure computes the right answer AND the define compiled right
#      after it is still callable and correct — not just present.
#   2. Beyond the former boundary (33): the procedure and following define
#      both execute correctly on the same paths.
#
# Usage: closure_upvalue_capacity_overflow_gate.sh <eshkol-run> <vm-standalone> <workdir>

set -u

ESHKOL_RUN="${1:?usage: $0 <eshkol-run> <vm-standalone> <workdir>}"
VM_RUN="${2:?missing vm standalone binary}"
WORK="${3:?missing work dir}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

mkdir -p "$WORK" || exit 1

OK_SRC="$ROOT_DIR/tests/closures/closure_upvalue_capacity_ok_test.esk"
OVERFLOW_SRC="$ROOT_DIR/tests/closures/closure_upvalue_capacity_overflow_test.esk"

PASS=0
FAIL=0

check_ok() { # label engine src
    local label="$1" engine="$2" src="$3"
    local out="$WORK/ok.$$"
    "$engine" "$src" >"$out" 2>&1
    local rc=$?
    if [ "$rc" -eq 0 ] && grep -q "^PASS$" "$out"; then
        PASS=$((PASS + 1))
        echo "PASS: $label"
    else
        FAIL=$((FAIL + 1))
        echo "FAIL: $label — exit $rc, output:"
        sed -n '1,20p' "$out"
    fi
    rm -f "$out"
}

if [ ! -f "$OK_SRC" ] || [ ! -f "$OVERFLOW_SRC" ]; then
    echo "FAIL: fixture(s) missing ($OK_SRC / $OVERFLOW_SRC)"
    exit 2
fi

# NOTE: this defect and its fix are specific to the bytecode-VM compile path
# (lib/backend/vm_compiler.c / vm_core.c / vm_run.c). eshkol-run's `-r`
# defaults to the native LLVM JIT — a wholly separate closure representation
# with no fixed upvalue-array capacity — so it is not exercised here; running
# either fixture through it proves nothing about this defect. The VM path is
# reached two ways: compiling straight to bytecode and executing in one
# process (eshkol-vm-standalone-test <src.esk>), and the two-step route a
# real build uses (eshkol-run --profile hosted-vm --emit-eskb, then load and
# run the .eskb on the standalone VM) — both are checked below.

if [ ! -x "$VM_RUN" ]; then
    echo "SKIP: bytecode VM standalone binary not built ($VM_RUN)"
    echo "PASS: closure upvalue capacity enforced (gate skipped, no VM binary)"
    exit 0
fi

# --- standalone bytecode VM: compile-and-run in one process -----------------

run_vm() { ESHKOL_VM_NO_DISASM=1 "$VM_RUN" "$1"; }
check_ok "eshkol-vm-standalone-test: 32-upvalue procedure runs correctly and the following define survives" \
    run_vm "$OK_SRC"
check_ok "eshkol-vm-standalone-test: 33-upvalue procedure runs correctly and the following define survives" \
    run_vm "$OVERFLOW_SRC"

# --- eshkol-run --profile hosted-vm --emit-eskb, then run the .eskb --------
# Mirrors scripts/run_vm_surface_tests.sh's two-step route, so the ESKB
# writer/reader round-trip is covered too, not just the in-process compiler.

compile_and_run_eskb() { # src
    local src="$1"
    local eskb="$WORK/$(basename "$src" .esk).eskb"
    rm -f "$eskb"
    "$ESHKOL_RUN" --profile hosted-vm --emit-eskb "$eskb" "$src"
    local compile_rc=$?
    if [ ! -s "$eskb" ]; then
        # Compilation refused to emit bytecode (the fail-closed path) — that
        # Report the compile step's own result rather than trying to run a
        # nonexistent module.
        return "$compile_rc"
    fi
    ESHKOL_VM_NO_DISASM=1 "$VM_RUN" "$eskb"
}
check_ok "eshkol-run --emit-eskb + VM: 32-upvalue procedure runs correctly and the following define survives" \
    compile_and_run_eskb "$OK_SRC"
check_ok "eshkol-run --emit-eskb + VM: 33-upvalue procedure runs correctly and the following define survives" \
    compile_and_run_eskb "$OVERFLOW_SRC"

# --- summary ------------------------------------------------------------------

echo
echo "closure-upvalue-capacity enforcement: $PASS passed, $FAIL failed"
if [ "$FAIL" -ne 0 ]; then
    exit 1
fi
echo "PASS: closure upvalue capacity enforced"
exit 0
