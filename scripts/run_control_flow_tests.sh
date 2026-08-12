#!/usr/bin/env bash

# Eshkol Control Flow Test Suite
# Runs all control flow tests and reports results

set -e

# Per-run, per-repo-root isolation for temp files and build artifacts.
# Two suites (two worktrees, two agents, CI plus a local run) must never share
# a scratch path or a build artifact — see scripts/lib/test_isolation.sh.
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
eshkol_test_isolation_init "control-flow"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASS=0
FAIL=0

echo "========================================="
echo "  Eshkol Control Flow Test Suite"
echo "========================================="
echo ""

# Honour $BUILD_DIR (CI passes it via the matrix); fall back to "build" for plain local runs.
BUILD_DIR="${BUILD_DIR:-build}"

# Some O0 AOT control-flow tests have large generated stack frames. Raise the
# stack limit for child test binaries where the host allows it, so the harness
# checks generated behavior instead of the caller shell's small default stack.
if ! ulimit -s unlimited 2>/dev/null; then
    ulimit -s 65532 2>/dev/null || true
fi

# Ensure build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: build directory not found. Run cmake first.${NC}"
    exit 1
fi

# Check if compiler exists
if [ ! -f "$BUILD_DIR/eshkol-run" ]; then
    echo -e "${RED}Error: eshkol-run not found. Run make first.${NC}"
    exit 1
fi

# Create directory if needed
mkdir -p tests/control_flow

echo "Testing all files in tests/control_flow/ directory..."
echo ""

# Run each test
for test_file in tests/control_flow/*.esk; do
    if [ ! -f "$test_file" ]; then
        echo "No test files found in tests/control_flow/"
        exit 0
    fi

    test_name=$(basename "$test_file")
    printf "Testing %-50s " "$test_name"

    # Clean up stale temp files before each test
    eshkol_test_reset_bin
    # Compile and run the test
    if ./$BUILD_DIR/eshkol-run -L./$BUILD_DIR "$test_file" -o "$ESHKOL_TEST_BIN" > /dev/null 2>&1; then
        if "$ESHKOL_TEST_BIN" > "$ESHKOL_TEST_OUT" 2>&1; then
            # Check for failures in output
            # A failure marker anywhere in the output fails the test — the old
            # `^FAIL`-anchored match never saw the indented `  <case>: FAIL`
            # form that most test programs actually print.
            if eshkol_test_output_has_failure "$ESHKOL_TEST_OUT"; then
                echo -e "${RED}❌ TESTS FAILED${NC}"
                grep "FAIL:" "$ESHKOL_TEST_OUT"
                ((FAIL++)) || true
            else
                echo -e "${GREEN}✅ PASS${NC}"
                ((PASS++)) || true
            fi
        else
            echo -e "${RED}❌ RUNTIME ERROR${NC}"
            ((FAIL++)) || true
        fi
    else
        echo -e "${RED}❌ COMPILE FAIL${NC}"
        ((FAIL++)) || true
    fi
done

echo ""
echo "========================================="
echo "  Test Results Summary"
echo "========================================="
echo -e "Total Tests:    $(( PASS + FAIL ))"
echo -e "${GREEN}Passed:         $PASS${NC}"
echo -e "${RED}Failed:         $FAIL${NC}"
echo ""

# Clean up
rm -f "$ESHKOL_TEST_OUT" "$ESHKOL_TEST_BIN"

# Exit with appropriate code
if [ $FAIL -eq 0 ]; then
    exit 0
else
    exit 1
fi
