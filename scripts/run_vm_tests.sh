#!/bin/bash
# Run the standalone bytecode VM self-tests.

set -e

BUILD_DIR="${BUILD_DIR:-build}"
VM="${BUILD_DIR}/eshkol-vm-standalone-test"

if [ ! -x "$VM" ]; then
    echo "eshkol-vm-standalone-test not found at $VM - build first." >&2
    exit 2
fi

OUT_TMP="$(mktemp "${TMPDIR:-/tmp}/eshkol_vm_standalone_tests.XXXXXX.log")"
cleanup_output() {
    rm -f -- "$OUT_TMP"
}
trap cleanup_output EXIT

echo "========================================="
echo "  Eshkol Bytecode VM Standalone Tests"
echo "========================================="
echo ""

set +e
ESHKOL_VM_NO_DISASM=1 "$VM" >"$OUT_TMP" 2>&1
rc=$?
set -e

if [ "$rc" -eq 0 ]; then
    if grep -q "Source tests: " "$OUT_TMP"; then
        grep "Source tests: " "$OUT_TMP" | tail -1
    fi
    echo ""
    echo "Passed: 1"
    echo "Failed: 0"
    exit 0
fi

tail -80 "$OUT_TMP"
echo ""
echo "Passed: 0"
echo "Failed: 1"
exit "$rc"
