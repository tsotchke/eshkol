#!/usr/bin/env bash
# `(process-spawn-pty cmd)` returns (pid . master-fd) on every engine, and a
# program that spawns many pseudo-terminal children in a row keeps working,
# because each master descriptor is handed to the program rather than retained
# by the runtime.

set -u

if [ "$#" -ne 3 ]; then
    echo "usage: $0 <eshkol-run> <eshkol-vm-standalone-test> <build-dir>" >&2
    exit 2
fi

ESH="$1"
VM="$2"
BUILD_DIR="$3"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PROBE="$ROOT_DIR/tests/capabilities/process_pty_handle_probe.esk"
WORK_DIR="$(mktemp -d "$BUILD_DIR/process-children.XXXXXX")"

failures=0

report_work_dir() {
    echo "NOTE: lane logs are under $WORK_DIR" >&2
}
trap report_work_dir EXIT

check_lane() {
    lane="$1"
    out="$WORK_DIR/$lane.out"
    for expected in "shape:pair" "pid-positive:yes" "fd-nonnegative:yes" "after-64:pair"; do
        if ! grep -q "$expected" "$out"; then
            echo "FAIL: $lane did not report $expected" >&2
            echo "  stdout: $(cat "$out")" >&2
            tail -20 "$WORK_DIR/$lane.err" >&2
            failures=$((failures + 1))
            return
        fi
    done
    echo "  ok  $lane"
}

ESHKOL_JIT_CACHE=0 "$ESH" -r "$PROBE" -L"$BUILD_DIR" \
    >"$WORK_DIR/jit.out" 2>"$WORK_DIR/jit.err"
check_lane jit

if "$ESH" "$PROBE" -o "$WORK_DIR/aot.bin" -L"$BUILD_DIR" >"$WORK_DIR/aot.build" 2>&1; then
    "$WORK_DIR/aot.bin" >"$WORK_DIR/aot.out" 2>"$WORK_DIR/aot.err"
    check_lane aot
else
    echo "FAIL: ahead-of-time compile did not succeed" >&2
    tail -40 "$WORK_DIR/aot.build" >&2
    failures=$((failures + 1))
fi

ESHKOL_VM_NO_DISASM=1 "$VM" "$PROBE" >"$WORK_DIR/vm.out" 2>"$WORK_DIR/vm.err"
check_lane vm

if [ "$failures" -ne 0 ]; then
    echo "FAIL: process_child_bookkeeping_test" >&2
    exit 1
fi
echo "PASS: process_child_bookkeeping_test"
