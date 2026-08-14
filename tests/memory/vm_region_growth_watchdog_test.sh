#!/usr/bin/env bash
# tests/memory/vm_region_growth_watchdog_test.sh — SW-14 interim guard gate.
#
# WHAT SW-14 IS. The bytecode VM has no heap reclamation of ANY kind:
# `(with-region ...)` is the designated mechanism and is a pure pass-through
# there, because the VM heap has no escape evacuator to promote a kept value
# out with (docs/KNOWN_ISSUES.md, docs/reference/runtime/memory-model.md,
# lib/backend/vm_compiler.c::compile_form_with_region). Measured on this
# fixture at the branch point: peak RSS is IDENTICAL with and without the
# `with-region` wrapper — 1.503 GB either way — so the form buys the user
# nothing and said nothing about it. Correct answers, exit 0, no diagnostic:
# the definition of SILENT.
#
# WHAT THIS GATE PINS. Not reclamation — reclamation is a separate build item
# (a VM-heap escape evacuator; see the SW-14 entry in
# .icc/silent-wrong-ledger.yaml). This gate pins the INTERIM guard that
# converts the silence into a named diagnostic, and it is written so that it
# FAILS if that guard is ever removed or quietly downgraded:
#
#   1. loud-note      a region form on the VM announces, once, that it
#                     reclaims nothing here.
#   2. note-silenced  ESHKOL_VM_REGION_QUIET=1 suppresses that note (so the
#                     note is a diagnostic, not unavoidable noise).
#   3. loud-growth    crossing ESHKOL_VM_HEAP_BUDGET_MB names the growth,
#                     the budget and the cause on stderr.
#   4. fail-closed    ESHKOL_VM_HEAP_BUDGET_FATAL=1 turns that into a nonzero
#                     exit, so a lane can gate on it.
#   5. budget-off     ESHKOL_VM_HEAP_BUDGET_MB=0 disables the watchdog.
#   6. answers        the guard changes no answer: the same program prints the
#                     same value with the watchdog on, off and fatal-off.
#
# Usage: tests/memory/vm_region_growth_watchdog_test.sh
#   BUILD_DIR env var selects the build directory (default: build).
set -u
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) VM="$BUILD_DIR/eshkol-vm-standalone-test" ;;
    *)  VM="$REPO_ROOT/$BUILD_DIR/eshkol-vm-standalone-test" ;;
esac
if [ ! -x "$VM" ]; then
    echo "vm_region_growth_watchdog_test.sh: $VM not found — run \`cmake --build $BUILD_DIR\` first." >&2
    exit 2
fi

SRC="$REPO_ROOT/tests/memory/vm_region_growth_watchdog_test.esk"
if [ ! -f "$SRC" ]; then
    echo "vm_region_growth_watchdog_test.sh: $SRC not found." >&2
    exit 2
fi

# Same convention as the other memory gates (define_loop_flat_rss_aot_test.sh):
# an ephemeral working directory, removed by the trap on every exit path. Total
# footprint is a few KB of captured stdout/stderr.
WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-vm-region-watchdog.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT INT TERM

PASS=0
FAIL=0
check() { # name condition-description result(0=ok)
    if [ "$3" -eq 0 ]; then
        echo "PASSED tests/memory/vm_region_growth_watchdog_test.sh::$1"
        PASS=$((PASS + 1))
    else
        echo "FAILED tests/memory/vm_region_growth_watchdog_test.sh::$1 — $2"
        FAIL=$((FAIL + 1))
    fi
}

# Every compared run sets ESHKOL_VM_NO_DISASM=1 so stdout is the program's own
# output rather than the VM's disassembly dump; reduce it to the digits printed.
export ESHKOL_VM_NO_DISASM=1
answer_of() { tr -cd '0-9' < "$1"; }

echo "  SW-14 interim guard: VM region growth watchdog"

# ── 1/2. the region note, and its silencer ────────────────────────────────
ESHKOL_VM_HEAP_BUDGET_MB=0 "$VM" "$SRC" >"$WORK/note.out" 2>"$WORK/note.err"
grep -q "reclaim no memory on the bytecode VM" "$WORK/note.err"
check loud_note "a VM region form did not announce that it reclaims nothing" $?

ESHKOL_VM_HEAP_BUDGET_MB=0 ESHKOL_VM_REGION_QUIET=1 \
    "$VM" "$SRC" >"$WORK/quiet.out" 2>"$WORK/quiet.err"
if grep -q "reclaim no memory on the bytecode VM" "$WORK/quiet.err"; then rc=1; else rc=0; fi
check note_silenced "ESHKOL_VM_REGION_QUIET=1 did not suppress the region note" $rc

# ── 3. the growth watchdog speaks ─────────────────────────────────────────
ESHKOL_VM_HEAP_BUDGET_MB=64 "$VM" "$SRC" >"$WORK/budget.out" 2>"$WORK/budget.err"
BUDGET_RC=$?
grep -q "heap budget exceeded" "$WORK/budget.err"
check loud_growth "crossing ESHKOL_VM_HEAP_BUDGET_MB produced no diagnostic" $?

grep -q "does not reclaim heap memory" "$WORK/budget.err"
check names_cause "the budget diagnostic does not name the VM's missing reclamation" $?

if [ "$BUDGET_RC" -eq 0 ]; then rc=0; else rc=1; fi
check advisory_by_default "the budget diagnostic is fatal without ESHKOL_VM_HEAP_BUDGET_FATAL" $rc

# ── 4. fail-closed mode ───────────────────────────────────────────────────
ESHKOL_VM_HEAP_BUDGET_MB=64 ESHKOL_VM_HEAP_BUDGET_FATAL=1 \
    "$VM" "$SRC" >"$WORK/fatal.out" 2>"$WORK/fatal.err"
FATAL_RC=$?
if [ "$FATAL_RC" -ne 0 ]; then rc=0; else rc=1; fi
check fail_closed "ESHKOL_VM_HEAP_BUDGET_FATAL=1 still exited 0 past the budget" $rc

# ── 5. the watchdog can be turned off ─────────────────────────────────────
if grep -q "heap budget exceeded" "$WORK/note.err"; then rc=1; else rc=0; fi
check budget_off "ESHKOL_VM_HEAP_BUDGET_MB=0 did not disable the watchdog" $rc

# ── 6. the guard changes no answer ────────────────────────────────────────
A_OFF="$(answer_of "$WORK/note.out")"
A_QUIET="$(answer_of "$WORK/quiet.out")"
A_BUDGET="$(answer_of "$WORK/budget.out")"
if [ -n "$A_OFF" ] && [ "$A_OFF" = "$A_QUIET" ] && [ "$A_OFF" = "$A_BUDGET" ]; then rc=0; else rc=1; fi
check answers_unchanged "the guard changed the program's output ('$A_OFF' / '$A_QUIET' / '$A_BUDGET')" $rc

echo "  vm-region-watchdog: $PASS passed, $FAIL failed"
if [ "$FAIL" -eq 0 ]; then
    echo "vm_region_growth_watchdog_test.sh: PASS"
    exit 0
fi
echo "vm_region_growth_watchdog_test.sh: FAIL"
exit 1
