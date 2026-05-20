#!/bin/bash
# Run the standalone bytecode VM self-tests.

set -e

BUILD_DIR="${BUILD_DIR:-build}"
VM="${BUILD_DIR}/eshkol-vm-standalone-test"

if [ ! -x "$VM" ]; then
    echo "eshkol-vm-standalone-test not found at $VM - build first." >&2
    exit 2
fi

OUT="${TMPDIR:-/tmp}/eshkol_vm_standalone_tests.log"

echo "========================================="
echo "  Eshkol Bytecode VM Standalone Tests"
echo "========================================="
echo ""

ESHKOL_VM_NO_DISASM=1 "$VM" >"$OUT" 2>&1
rc=$?

if [ "$rc" -eq 0 ]; then
    if grep -q "Source tests: " "$OUT"; then
        grep "Source tests: " "$OUT" | tail -1
    fi
    echo ""
    echo "Passed: 1"
    echo "Failed: 0"
    exit 0
fi

tail -80 "$OUT"
echo ""
echo "Passed: 0"
echo "Failed: 1"
exit "$rc"
