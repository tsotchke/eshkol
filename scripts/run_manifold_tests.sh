#!/usr/bin/env bash
# Compile and run the differential-geometry regression suite.

set -euo pipefail

# Per-run, per-repo-root isolation and the shared honest-detection helpers.
ESHKOL_TEST_ISOLATION_NO_TRAP=1
# Sourcing must be checked *before* the fact: bash 3.2 (macOS) exits the
# shell when `source` cannot find its file, so a trailing `|| {...}` never
# runs there. A suite with no prelude has no failure detection and no
# scratch isolation, and must refuse to run rather than report a PASS.
ESHKOL_TEST_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/test_isolation.sh"
if [ ! -r "$ESHKOL_TEST_LIB" ]; then
    echo "FATAL: cannot read $ESHKOL_TEST_LIB" >&2
    echo "       (the shared test isolation and failure-detection prelude)." >&2
    echo "       Refusing to run: without it this suite would report a" >&2
    echo "       meaningless PASS." >&2
    exit 2
fi
source "$ESHKOL_TEST_LIB"
eshkol_test_isolation_init "manifold"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-build}"
ESHKOL_RUN="$ROOT_DIR/$BUILD_DIR/eshkol-run"
TEST_FILE="$ROOT_DIR/tests/manifold/manifold_test.esk"
RUN_DIR="$ESHKOL_TEST_TMPDIR/run"
mkdir -p "$RUN_DIR"
TEST_BIN="$RUN_DIR/manifold_test"
RUN_OUT="$RUN_DIR/manifold_test.out"
cleanup() { eshkol_test_isolation_cleanup; }
trap cleanup EXIT

echo "========================================="
echo "  Eshkol Differential Geometry Tests"
echo "========================================="

if [ ! -x "$ESHKOL_RUN" ]; then
    echo "eshkol-run not found at $ESHKOL_RUN - build first." >&2
    exit 2
fi

"$ESHKOL_RUN" -L"$ROOT_DIR/$BUILD_DIR" "$TEST_FILE" -o "$TEST_BIN"

# The summary below used to be hardcoded "Passed: 1 / Failed: 0", printed
# whenever the binary exited 0. The test program prints its own per-case
# verdicts and exits 0 regardless, so a failing regression was reported as a
# pass — and run_all_tests.sh scraped that fabricated summary into the
# aggregate count. Derive the verdict from what the program actually printed.
set +e
"$TEST_BIN" > "$RUN_OUT" 2>&1
run_rc=$?
set -e
cat "$RUN_OUT"

if [ "$run_rc" -ne 0 ]; then
    echo ""
    echo "Passed: 0"
    echo "Failed: 1"
    exit "$run_rc"
fi

if eshkol_test_output_has_failure "$RUN_OUT" 'error:'; then
    echo ""
    echo "Failing lines:"
    eshkol_test_output_failures "$RUN_OUT" 'error:' 20 | sed 's/^/  /'
    echo ""
    echo "Passed: 0"
    echo "Failed: 1"
    exit 1
fi

if eshkol_test_output_is_silent "$RUN_OUT"; then
    echo ""
    echo "The manifold test produced no output — absence of a verdict is not a pass."
    echo "Passed: 0"
    echo "Failed: 1"
    exit 1
fi

echo ""
echo "Passed: 1"
echo "Failed: 0"
