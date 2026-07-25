#!/bin/bash
# Run the standalone bytecode VM self-tests.

set -e

BUILD_DIR="${BUILD_DIR:-build}"
VM="${BUILD_DIR}/eshkol-vm-standalone-test"

if [ ! -x "$VM" ]; then
    echo "eshkol-vm-standalone-test not found at $VM - build first." >&2
    exit 2
fi

# Per-run, per-repo-root isolation and the shared honest-detection helpers.
ESHKOL_TEST_ISOLATION_NO_TRAP=1
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/test_isolation.sh"
eshkol_test_isolation_init "vm"

OUT_TMP="$ESHKOL_TEST_TMPDIR/vm_standalone_tests.log"
cleanup_output() {
    eshkol_test_isolation_cleanup
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
    # A zero exit status alone is not a pass.
    #
    # Two ways this used to certify a broken VM: the self-test prints failures
    # and still exits 0, and the expected "Source tests: N/N" line was only
    # echoed *if present* — its absence (a VM that died before reaching the
    # source-test phase, or printed nothing) fell through to the hardcoded
    # "Passed: 1 / Failed: 0". Require the marker and reject failure markers.
    if eshkol_test_output_has_failure "$OUT_TMP"; then
        tail -80 "$OUT_TMP"
        echo ""
        echo "Failing lines:"
        eshkol_test_output_failures "$OUT_TMP" "" 20 | sed 's/^/  /'
        echo ""
        echo "Passed: 0"
        echo "Failed: 1"
        exit 1
    fi

    if ! eshkol_test_output_has_marker "$OUT_TMP" 'Source tests: '; then
        tail -80 "$OUT_TMP"
        echo ""
        echo "Expected 'Source tests: ' summary is absent — the VM self-test did"
        echo "not reach the end of its run. Absence of a verdict is not a pass."
        echo "Passed: 0"
        echo "Failed: 1"
        exit 1
    fi

    grep "Source tests: " "$OUT_TMP" | tail -1
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
