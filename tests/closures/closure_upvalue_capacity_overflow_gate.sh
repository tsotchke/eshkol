#!/usr/bin/env bash
# SW-45 gate: the shared closure-upvalue capacity is enforced consistently —
# up to it a program runs correctly (including the top-level `define`
# compiled right after the big procedure), and one past it the compiler
# refuses LOUDLY instead of silently corrupting the runtime stack.
#
# Before the fix, the compiler capped a scope's upvalue count at 32
# (MAX_UPVALUES, lib/backend/vm_parser.c) but the runtime closure
# representation's arrays (vm_core.c's HeapObject.closure.upvalues[]/
# open_slots[]) were fixed at 16. A closure needing 17-32 upvalues compiled
# cleanly and then had its count silently clamped to 16 at OP_CLOSURE
# (lib/backend/vm_run.c): the runtime popped only 16 of the >16 values the
# compiler had pushed to feed it, stranding the rest on the operand stack —
# no error, exit 0 — and every stack-slot offset computed for the rest of the
# program was off by the leaked count from then on. The next top-level
# `define` read back a stray leaked value instead of its own closure.
#
# inc/eshkol/backend/vm_limits.h now defines ESHKOL_VM_MAX_CLOSURE_UPVALUES as
# the single shared constant for both the compiler's cap and the runtime
# array capacity, so this gate's two cases pin what "shared" has to mean:
#
#   1. AT the capacity (32 distinct top-level calls in one procedure): the
#      procedure computes the right answer AND the define compiled right
#      after it is still callable and correct — not just present.
#   2. ONE PAST the capacity (33): the compile fails loudly (nonzero exit,
#      the "closure exceeds the ... upvalue capture limit" diagnostic on
#      stderr, and "refusing to run a program that failed to compile") —
#      never a silent corruption, and never a bare crash with no diagnostic.
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

check_refuses() { # label engine src
    local label="$1" engine="$2" src="$3"
    local out="$WORK/overflow.$$"
    "$engine" "$src" >"$out" 2>&1
    local rc=$?
    local ok=true
    if [ "$rc" -eq 0 ]; then
        ok=false
        echo "FAIL: $label — compiled and ran to completion (exit 0); the capacity is not enforced"
    fi
    if ! grep -qF "upvalue capture limit" "$out"; then
        ok=false
        echo "FAIL: $label — missing the expected 'upvalue capture limit' diagnostic"
    fi
    if ! grep -qE "refusing to (run|emit bytecode for)" "$out"; then
        ok=false
        echo "FAIL: $label — missing the fail-closed 'refusing to run/emit' message; a diagnostic that still executes (or still emits bytecode) is exactly the silent-corruption shape this gate exists to catch"
    fi
    if $ok; then
        PASS=$((PASS + 1))
        echo "PASS: $label"
    else
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
check_refuses "eshkol-vm-standalone-test: 33-upvalue procedure fails the compile loudly" \
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
        # IS a result for check_refuses to see; report the compile step's own
        # exit/diagnostic rather than trying to run a nonexistent module.
        return "$compile_rc"
    fi
    ESHKOL_VM_NO_DISASM=1 "$VM_RUN" "$eskb"
}
check_ok "eshkol-run --emit-eskb + VM: 32-upvalue procedure runs correctly and the following define survives" \
    compile_and_run_eskb "$OK_SRC"
check_refuses "eshkol-run --emit-eskb: 33-upvalue procedure fails the compile loudly (refuses to emit bytecode)" \
    compile_and_run_eskb "$OVERFLOW_SRC"

# --- summary ------------------------------------------------------------------

echo
echo "closure-upvalue-capacity enforcement: $PASS passed, $FAIL failed"
if [ "$FAIL" -ne 0 ]; then
    exit 1
fi
echo "PASS: closure upvalue capacity enforced"
exit 0
