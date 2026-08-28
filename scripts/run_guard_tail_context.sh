#!/usr/bin/env bash
# scripts/run_guard_tail_context.sh — SW-58 value-differential gate.
#
# tests/tco/guard_tail_context/ asks one question by measurement: when a
# self-recursive tail call sits in a `guard` body and Eshkol lowers it as a loop
# back edge, WHICH HANDLER ANSWERS?  That is not a stack question — it costs
# nothing and shows up only as a different value — so scripts/run_tco_tests.sh,
# which gates on a clean exit, cannot see it.  This runner gates on the VALUE.
#
# The reference answers below were measured on chibi-scheme 0.12, an R7RS
# reference implementation, running each fixture unchanged apart from an
# `(import (scheme base) (scheme write))` prologue.  They are recorded here and
# in the directory's README; re-measure with the loop at the bottom of that
# README rather than by editing a number.
#
# Every fixture is checked on all three engines Eshkol ships — native JIT (-r),
# native AOT, and the bytecode VM — because SW-58 was an engine-local defect:
# the VM never treated a `guard` body as a tail position, so it answered
# correctly while both native paths did not.  Parity across the three is the
# property being gated, and the reference is what all three are held to.
#
# Usage: scripts/run_guard_tail_context.sh
#   BUILD_DIR selects the build directory (default: build).
set -u

BUILD_DIR="${BUILD_DIR:-build}"
ESHKOL_RUN="${ESHKOL_RUN:-$BUILD_DIR/eshkol-run}"
ESHKOL_VM="${ESHKOL_VM:-$BUILD_DIR/eshkol-vm-standalone-test}"
FIXTURES="tests/tco/guard_tail_context"

if [ ! -x "$ESHKOL_RUN" ]; then
    echo "run_guard_tail_context.sh: $ESHKOL_RUN not found — build eshkol-run first." >&2
    exit 1
fi

# fixture basename <TAB> reference answer (chibi-scheme 0.12)
read -r -d '' EXPECTED <<'TABLE'
01_outer_guard_catches_callee_raise	(caught-by-a 0)
02_innermost_guard_wins	(gb 0)
03_nested_guards_inner_answers	inner
04_reraise_reaches_enclosing_guard	(inner 1)
05_reraise_chain_walks_out_one_activation_at_a_time	(answered-at 0)
06_clause_raise_reaches_previous_activation	(bottom (1 2 3 . boom))
07_nested_guards_reraise_through_both	(outer-answered 1 (i o i o i . boom))
08_guard_dynamic_wind_order	((before 0) (before 1) (before 2) (after 2) (handler 2) (after 1) (handler 1) (after 0) (caught 0))
09_clause_reads_a_binding_the_loop_rebinds	(inner 0 0)
TABLE

WORK="$(mktemp -d "${TMPDIR:-/tmp}/guard_tail_context.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

pass=0
fail=0
failed=()

check() {
    local engine="$1" name="$2" want="$3" got="$4"
    if [ "$got" = "$want" ]; then
        printf '  %-6s %-54s PASS\n' "$engine" "$name"
        pass=$((pass + 1))
    else
        printf '  %-6s %-54s FAIL\n' "$engine" "$name"
        printf '         got:  %s\n' "$got"
        printf '         want: %s\n' "$want"
        fail=$((fail + 1))
        failed+=("$engine/$name")
    fi
}

echo "========================================="
echo "  SW-58 guard tail-context differential"
echo "  reference: chibi-scheme 0.12"
echo "========================================="

while IFS=$'\t' read -r name want; do
    [ -n "$name" ] || continue
    src="$FIXTURES/$name.esk"
    if [ ! -f "$src" ]; then
        echo "  MISSING FIXTURE $src" >&2
        fail=$((fail + 1))
        failed+=("missing/$name")
        continue
    fi

    got=$("$ESHKOL_RUN" -r "$src" 2>/dev/null | tail -1)
    check "jit" "$name" "$want" "$got"

    bin="$WORK/$name.bin"
    if "$ESHKOL_RUN" "$src" -o "$bin" >/dev/null 2>&1; then
        got=$("$bin" 2>/dev/null | tail -1)
        check "aot" "$name" "$want" "$got"
    else
        printf '  %-6s %-54s FAIL (compile)\n' "aot" "$name"
        fail=$((fail + 1))
        failed+=("aot/$name")
    fi

    if [ -x "$ESHKOL_VM" ]; then
        got=$("$ESHKOL_VM" "$src" 2>/dev/null | tail -1)
        check "vm" "$name" "$want" "$got"
    fi
done <<< "$EXPECTED"

echo ""
echo "Passed: $pass  Failed: $fail"
if [ "$fail" -ne 0 ]; then
    printf 'FAILED: %s\n' "${failed[@]}"
    echo "run_guard_tail_context.sh: FAIL"
    exit 1
fi
echo "run_guard_tail_context.sh: PASS"
