#!/usr/bin/env bash
# ESH-0216: the runtime_shutdown_teardown_race_test scenario (SIGTERM while
# the global thread pool is busy and a shutdown hook frees shared arena
# state) is inherently racy — a single clean run doesn't prove the ordering
# fix holds. Run the binary CYCLES times (mandated: 50) and fail if any
# iteration reports a fatal signal or dies from an unexpected one.
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/${BUILD_DIR:-build}/runtime_shutdown_teardown_race_test"
CYCLES="${1:-50}"

if [ ! -x "$BIN" ]; then
    echo "SKIP: $BIN not built"
    exit 0
fi

WORK=$(mktemp -d -t eshkol_shutdown_race.XXXXXX)
trap 'rm -rf "$WORK"' EXIT

fail_count=0
for i in $(seq 1 "$CYCLES"); do
    LOG="$WORK/cycle_${i}.log"
    "$BIN" >"$LOG" 2>&1
    rc=$?

    if grep -q "fatal signal" "$LOG"; then
        echo "FAIL: cycle $i hit a fatal signal (rc=$rc)"
        cat "$LOG"
        fail_count=$((fail_count + 1))
        continue
    fi

    if [ "$rc" -ne 0 ]; then
        echo "FAIL: cycle $i exited rc=$rc (expected 0)"
        cat "$LOG"
        fail_count=$((fail_count + 1))
        continue
    fi
done

if [ "$fail_count" -ne 0 ]; then
    echo "RESULT: FAIL - $fail_count/$CYCLES cycles hit the teardown race"
    exit 1
fi

echo "RESULT: PASS - $CYCLES/$CYCLES cycles clean, no SIGSEGV during shutdown"
