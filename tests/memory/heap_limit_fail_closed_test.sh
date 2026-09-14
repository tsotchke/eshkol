#!/usr/bin/env bash
# tests/memory/heap_limit_fail_closed_test.sh — SW-165: the heap ceiling is a
# fail-closed contract, and its unit is the documented one.
#
# ESHKOL_MAX_HEAP is documented (docs/reference/runtime/environment-variables.md)
# as a byte count accepting a K/M/G suffix. Three things were wrong with what
# the runtime actually did:
#
#   * Heap accounting — a function called for every arena block — printed the
#     breach itself, so a run that crossed the ceiling printed the same line for
#     as long as it kept running.
#   * It printed on the DEFAULT ceiling, which no user asked for.
#   * The interrupt it requested named a memory shutdown, but the only reader of
#     that flag acts on a timeout and returns for every other reason. Nothing
#     ever acted on it: the process announced the breach and then exited 0.
#
# A ceiling that is loud but does not bind is worse than either a silent one or
# an enforced one, because it teaches a reader to ignore it. This gate pins the
# contract in both directions:
#
#   1. No ceiling asked for  -> runs to completion, exit 0, NOTHING on stderr.
#   2. Ceiling that binds    -> exactly one fatal line, nonzero exit, no
#                               COMPLETED, and the ceiling reported in the unit
#                               it was given (a sub-megabyte ceiling must not
#                               print as "0MB").
#   3. Ceiling with room     -> runs to completion, exit 0, no warnings.
#   4. Malformed value       -> says so on stderr, naming the variable and the
#                               accepted grammar, and falls back to the default
#                               rather than leaving the operator to believe a
#                               bound is in force that is not.
#   5. Suffixes parse as documented: 1K/1KiB/1KB all mean 1024 bytes.
#
# Cases 1, 3 and 5 are the negative controls: they are what makes case 2
# evidence of enforcement rather than of a runtime that fails everything.
#
# Usage: tests/memory/heap_limit_fail_closed_test.sh
#   BUILD_DIR env var selects the build directory (default: build).
#   ESHKOL_RUN env var overrides the eshkol-run binary path directly.
set -u
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"

BUILD_DIR="${BUILD_DIR:-build}"
if [ -z "${ESHKOL_RUN:-}" ]; then
    case "$BUILD_DIR" in
        /*) ESHKOL_RUN="$BUILD_DIR/eshkol-run" ;;
        *) ESHKOL_RUN="$REPO_ROOT/$BUILD_DIR/eshkol-run" ;;
    esac
fi
if [ ! -x "$ESHKOL_RUN" ]; then
    echo "heap_limit_fail_closed_test.sh: $ESHKOL_RUN not found — run \`cmake --build $BUILD_DIR --target eshkol-run stdlib\` first." >&2
    exit 2
fi

SRC="$REPO_ROOT/tests/memory/heap_limit_fail_closed_test.esk"
[ -f "$SRC" ] || { echo "heap_limit_fail_closed_test.sh: $SRC not found." >&2; exit 2; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/eshkol-heap-limit.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

echo "=========================================================="
echo "  SW-165 heap-limit fail-closed contract"
echo "=========================================================="
echo

BIN="$WORK/heap_limit_bin"
( cd "$WORK" && ESHKOL_PATH="$REPO_ROOT/lib" "$ESHKOL_RUN" "$SRC" -o "$BIN" ) \
    > "$WORK/compile.log" 2>&1
if [ $? -ne 0 ]; then
    echo "FAIL: AOT compile failed. Output:"
    cat "$WORK/compile.log"
    echo "heap_limit_fail_closed_test.sh: FAIL"
    exit 1
fi
chmod +x "$BIN"

fail=0
note() { echo "      -- $*"; }

# run_with <tag> <env-assignment-or-empty>  -> sets RC, OUT, ERR file paths
run_with() {
    tag="$1"; shift
    OUT="$WORK/$tag.out"; ERR="$WORK/$tag.err"
    if [ $# -gt 0 ] && [ -n "$1" ]; then
        env "$@" "$BIN" > "$OUT" 2> "$ERR"
    else
        "$BIN" > "$OUT" 2> "$ERR"
    fi
    RC=$?
    ERR_LINES=$(wc -l < "$ERR" | tr -d ' ')
}

# ---- 1. no ceiling asked for: completes, silent -----------------------------
run_with default
if [ "$RC" -ne 0 ] || ! grep -q "^COMPLETED$" "$OUT"; then
    echo "FAIL [default]: the run did not complete (exit=$RC)."
    note "with no ESHKOL_MAX_HEAP set, the default ceiling is an accounting"
    note "reference, not a budget — it must never stop a run."
    head -5 "$ERR"
    fail=1
elif [ "$ERR_LINES" -ne 0 ]; then
    echo "FAIL [default]: $ERR_LINES line(s) on stderr with no ceiling requested."
    note "this is the spam the fix removed: heap accounting must not diagnose."
    head -5 "$ERR"
    fail=1
else
    echo "  [default]        exit=0, completed, stderr silent"
fi

# ---- 2. a ceiling that binds: one fatal line, nonzero exit ------------------
for CEILING in 512 1M; do
    run_with "bind_$CEILING" "ESHKOL_MAX_HEAP=$CEILING"
    if [ "$RC" -eq 0 ]; then
        echo "FAIL [ESHKOL_MAX_HEAP=$CEILING]: exited 0 despite exceeding the ceiling."
        note "the ceiling must FAIL CLOSED — announcing a breach and continuing"
        note "to a zero exit status is the defect this gate exists for."
        fail=1
        continue
    fi
    if grep -q "^COMPLETED$" "$OUT"; then
        echo "FAIL [ESHKOL_MAX_HEAP=$CEILING]: the workload ran to completion anyway."
        fail=1
        continue
    fi
    # Exactly one FATAL line. A ceiling that is actually approached may also
    # emit its one-shot soft-limit warning first, which is the design and not
    # noise — so the assertion is on the fatal count and on the total staying
    # bounded, which is what distinguishes "reported once" from "reported for
    # the rest of the run".
    FATAL_LINES=$(grep -c "fatal:" "$ERR")
    if [ "$FATAL_LINES" -ne 1 ] || [ "$ERR_LINES" -gt 2 ]; then
        echo "FAIL [ESHKOL_MAX_HEAP=$CEILING]: expected exactly 1 fatal line (at most 1 soft-limit warning alongside it); got $FATAL_LINES fatal of $ERR_LINES total."
        note "the breach must be reported ONCE, at the one site that acts on it,"
        note "not once per allocation for the rest of the run."
        head -4 "$ERR"
        fail=1
        continue
    fi
    if ! grep -q "ESHKOL_MAX_HEAP" "$ERR"; then
        echo "FAIL [ESHKOL_MAX_HEAP=$CEILING]: the diagnostic does not name the variable that set the limit."
        cat "$ERR"
        fail=1
        continue
    fi
    if grep -q "0MB" "$ERR"; then
        echo "FAIL [ESHKOL_MAX_HEAP=$CEILING]: the ceiling is reported as \"0MB\"."
        note "integer-MB formatting erased every sub-megabyte ceiling; the"
        note "diagnostic must report the ceiling in the unit it was given."
        cat "$ERR"
        fail=1
        continue
    fi
    echo "  [MAX_HEAP=$CEILING] exit=$RC, one fatal line, no COMPLETED"
done

# ---- 3. a ceiling with room: completes, no warnings (negative control) ------
run_with ample "ESHKOL_MAX_HEAP=4G"
if [ "$RC" -ne 0 ] || ! grep -q "^COMPLETED$" "$OUT"; then
    echo "FAIL [ESHKOL_MAX_HEAP=4G]: a ceiling with ample room stopped the run (exit=$RC)."
    note "without this control, a runtime that failed EVERY run would pass"
    note "the fail-closed case above."
    head -5 "$ERR"
    fail=1
elif [ "$ERR_LINES" -ne 0 ]; then
    echo "FAIL [ESHKOL_MAX_HEAP=4G]: $ERR_LINES line(s) on stderr well under the ceiling."
    head -5 "$ERR"
    fail=1
else
    echo "  [MAX_HEAP=4G]    exit=0, completed, stderr silent"
fi

# ---- 4. a malformed value: says so, then falls back ------------------------
# Reporting rather than refusing matches what ESHKOL_STACK_SIZE already does and
# what the three non-size limit variables do; what must never happen is the
# SILENT fallback, which left an operator believing a bound was in force.
for BAD in "512MBB" "not-a-size" "12X"; do
    run_with "bad" "ESHKOL_MAX_HEAP=$BAD"
    if ! grep -q "ESHKOL_MAX_HEAP" "$ERR"; then
        echo "FAIL [ESHKOL_MAX_HEAP=$BAD]: the bad value was accepted in silence."
        note "falling back without saying so means the operator believes a bound"
        note "is in force that is not — the whole point of this case."
        fail=1
    elif ! grep -q "K/M/G" "$ERR"; then
        echo "FAIL [ESHKOL_MAX_HEAP=$BAD]: the diagnostic does not name the accepted grammar."
        cat "$ERR"
        fail=1
    elif [ "$RC" -ne 0 ] || ! grep -q "^COMPLETED$" "$OUT"; then
        echo "FAIL [ESHKOL_MAX_HEAP=$BAD]: the run did not fall back to the default (exit=$RC)."
        cat "$ERR"
        fail=1
    else
        echo "  [MAX_HEAP=$BAD] reported and fell back to the default, exit=0"
    fi
done

# ---- 5. the documented suffix grammar --------------------------------------
# 1K, 1KiB and 1KB all mean 1024 bytes, so all three must bind identically.
suffix_fail=0
for SPELLING in 1K 1KiB 1KB; do
    run_with "suffix" "ESHKOL_MAX_HEAP=$SPELLING"
    if [ "$RC" -eq 0 ] || grep -q "is not a valid size" "$ERR"; then
        echo "FAIL [ESHKOL_MAX_HEAP=$SPELLING]: the documented suffix grammar did not parse (exit=$RC)."
        cat "$ERR"
        suffix_fail=1
        fail=1
    fi
done
[ "$suffix_fail" -eq 0 ] && echo "  [suffixes]       1K / 1KiB / 1KB all parse and all bind"

echo
if [ "$fail" -eq 0 ]; then
    echo "heap_limit_fail_closed_test.sh: PASS"
else
    echo "heap_limit_fail_closed_test.sh: FAIL"
fi
exit "$fail"
