#!/usr/bin/env bash
# The environment overlay `(process-spawn cmd env)` takes belongs to the child.
# The child's environment carries it; the spawning program's own environment is
# the same afterwards as it was before, so overlays do not accumulate across
# spawns.
#
# This matters most on Windows, where the child environment is a block passed
# to CreateProcess rather than an assignment inside a forked child — which is
# why the test runs on every platform's CI rather than on POSIX alone.

set -u

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <eshkol-run> <build-dir>" >&2
    exit 2
fi

ESH="$1"
BUILD_DIR="$2"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PROBE="$ROOT_DIR/tests/capabilities/process_child_env_probe.esk"
WORK_DIR="$(mktemp -d "$BUILD_DIR/process-child-env.XXXXXX")"

failures=0

report_work_dir() {
    echo "NOTE: lane logs are under $WORK_DIR" >&2
}
trap report_work_dir EXIT

if [ -n "${ESHKOL_CHILD_ENV_PROBE:-}" ]; then
    echo "FAIL: ESHKOL_CHILD_ENV_PROBE is already set in the environment" >&2
    exit 1
fi
# The parent holds this one, so the overlay has something to override.
export ESHKOL_CHILD_ENV_OVERRIDE=parent-value

check_lane() {
    lane="$1"
    out="$WORK_DIR/$lane.out"
    ok=1
    grep -q "spawned:yes" "$out" || { echo "FAIL: $lane did not spawn" >&2; ok=0; }
    grep -q "child-saw-overlay:yes" "$out" || {
        echo "FAIL: $lane child did not receive the overlay" >&2; ok=0; }
    grep -q "child-saw-override:yes" "$out" || {
        echo "FAIL: $lane overlay did not override the parent's own value" >&2; ok=0; }
    grep -q "parent-override-intact:yes" "$out" || {
        echo "FAIL: $lane the override changed the spawning program's value" >&2; ok=0; }
    grep -q "parent-unchanged:yes" "$out" || {
        echo "FAIL: $lane overlay reached the spawning program's environment" >&2; ok=0; }
    if [ "$ok" -eq 1 ]; then
        echo "  ok  $lane"
    else
        echo "  stdout: $(cat "$out")" >&2
        tail -20 "$WORK_DIR/$lane.err" >&2
        failures=$((failures + 1))
    fi
}

ESHKOL_CHILD_ENV_DUMP="$WORK_DIR/jit-dump.txt" ESHKOL_JIT_CACHE=0 \
    "$ESH" -r "$PROBE" -L"$BUILD_DIR" >"$WORK_DIR/jit.out" 2>"$WORK_DIR/jit.err"
check_lane jit

if "$ESH" "$PROBE" -o "$WORK_DIR/aot.bin" -L"$BUILD_DIR" >"$WORK_DIR/aot.build" 2>&1; then
    ESHKOL_CHILD_ENV_DUMP="$WORK_DIR/aot-dump.txt" \
        "$WORK_DIR/aot.bin" >"$WORK_DIR/aot.out" 2>"$WORK_DIR/aot.err"
    check_lane aot
else
    echo "FAIL: ahead-of-time compile did not succeed" >&2
    tail -40 "$WORK_DIR/aot.build" >&2
    failures=$((failures + 1))
fi

if [ "$failures" -ne 0 ]; then
    echo "FAIL: process_child_env_test" >&2
    exit 1
fi
echo "PASS: process_child_env_test"
