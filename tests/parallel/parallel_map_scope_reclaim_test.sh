#!/usr/bin/env bash
# tests/parallel/parallel_map_scope_reclaim_test.sh
#
# Concurrency regression gate: parallel-map over a closure whose body uses
# scope-based reclamation (internal named-let loops + memv) must not corrupt
# results by racing the shared arena's single scope stack across pool workers.
#
# A bump-arena scope stack is single-threaded; before the fix, concurrent
# workers pinned to the shared arena would push/pop/rewind it simultaneously,
# so one worker's pop freed memory another was using -> "car/cdr: argument is
# not a pair", SIGSEGV/SIGBUS, or a hang, nondeterministically above the
# parallel threshold. The fix degrades scope operations to commit-only on pool
# workers.
#
# The race is probabilistic, so this gate compiles the fixture AOT and runs it
# REPEATEDLY under ESHKOL_ARENA_POISON=1 (freed arena bytes are filled with
# 0xCB, turning a dangling read into a loud crash instead of a silent stale
# read). It fails if ANY run crashes, hangs, or does not print "RESULT: PASS".
#
# Usage: tests/parallel/parallel_map_scope_reclaim_test.sh [--runs N] [--timeout S]
#   BUILD_DIR env var selects the build directory (default: build).
#   ESHKOL_RUN env var overrides the eshkol-run binary path directly.
set -u
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"

RUNS=25
PER_RUN_TIMEOUT=60
while [ $# -gt 0 ]; do
    case "$1" in
        --runs) RUNS="$2"; shift 2 ;;
        --timeout) PER_RUN_TIMEOUT="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in
    /*) : ;;
    *)  BUILD_DIR="$REPO_ROOT/$BUILD_DIR" ;;
esac
ESHKOL_RUN="${ESHKOL_RUN:-$BUILD_DIR/eshkol-run}"

if [ ! -x "$ESHKOL_RUN" ]; then
    echo "parallel_map_scope_reclaim_test.sh: eshkol-run not found at $ESHKOL_RUN" >&2
    exit 2
fi

SRC="$REPO_ROOT/tests/parallel/parallel_map_scope_reclaim_test.esk"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
BIN="$WORK/scope_reclaim"

# Optional per-run timeout wrapper (coreutils timeout / gtimeout if present).
TIMEOUT_BIN=""
if command -v timeout >/dev/null 2>&1; then TIMEOUT_BIN="timeout"
elif command -v gtimeout >/dev/null 2>&1; then TIMEOUT_BIN="gtimeout"; fi

echo "parallel_map_scope_reclaim_test.sh: compiling AOT ..."
if ! "$ESHKOL_RUN" "$SRC" -o "$BIN" >"$WORK/compile.log" 2>&1; then
    echo "parallel_map_scope_reclaim_test.sh: FAIL (AOT compile failed)"
    tail -20 "$WORK/compile.log"
    exit 1
fi

echo "parallel_map_scope_reclaim_test.sh: running $RUNS iterations under ESHKOL_ARENA_POISON=1 ..."
for i in $(seq 1 "$RUNS"); do
    if [ -n "$TIMEOUT_BIN" ]; then
        out=$(ESHKOL_ARENA_POISON=1 "$TIMEOUT_BIN" "$PER_RUN_TIMEOUT" "$BIN" 2>&1); rc=$?
    else
        out=$(ESHKOL_ARENA_POISON=1 "$BIN" 2>&1); rc=$?
    fi
    if [ "$rc" -ne 0 ]; then
        echo "parallel_map_scope_reclaim_test.sh: FAIL (run $i crashed/timed out, exit $rc)"
        printf '%s\n' "$out" | tail -5
        exit 1
    fi
    if ! printf '%s\n' "$out" | grep -q "RESULT: PASS"; then
        echo "parallel_map_scope_reclaim_test.sh: FAIL (run $i did not print RESULT: PASS)"
        printf '%s\n' "$out" | tail -5
        exit 1
    fi
done

echo "parallel_map_scope_reclaim_test.sh: PASS ($RUNS/$RUNS runs clean under poison)"
exit 0
