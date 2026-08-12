#!/usr/bin/env bash

# Eshkol Signal Processing Test Suite
# Tests signal processing filters, windowing, and convolution

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
eshkol_test_isolation_init "signal"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASS=0
FAIL=0

# Results arrays
declare -a FAILED_TESTS

echo "========================================="
echo "  Eshkol Signal Processing Tests"
echo "========================================="
echo ""

# Check for build directory
BUILD_DIR="${BUILD_DIR:-build}"

if [ ! -d "$BUILD_DIR" ] || [ ! -f "$BUILD_DIR/eshkol-run" ]; then
    echo -e "${RED}Error: Build directory not found or eshkol-run missing.${NC}"
    echo "Please build first: cd build && cmake .. && make -j8"
    exit 1
fi

echo -e "${GREEN}Using build directory: $BUILD_DIR${NC}"
echo ""

# Check for test directory
SIGNAL_TEST_DIR="tests/signal"

if [ ! -d "$SIGNAL_TEST_DIR" ]; then
    echo -e "${RED}Error: $SIGNAL_TEST_DIR directory not found.${NC}"
    exit 1
fi

# Count test files
TEST_COUNT=$(find "$SIGNAL_TEST_DIR" -name "*.esk" 2>/dev/null | wc -l | tr -d ' ')
if [ "$TEST_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}No signal test files found (*.esk).${NC}"
    exit 0
fi

echo "Found $TEST_COUNT test file(s)"
echo ""

# Run each .esk test
for test_file in "$SIGNAL_TEST_DIR"/*.esk; do
    if [ ! -f "$test_file" ]; then
        continue
    fi

    test_name=$(basename "$test_file")
    printf "Testing %-45s " "$test_name"

    # Clean up stale artifacts
    eshkol_test_reset_bin
    # Try to compile
    if ./$BUILD_DIR/eshkol-run "$test_file" -L./$BUILD_DIR -o "$ESHKOL_TEST_BIN" > "$ESHKOL_TEST_COMPILE_LOG" 2>&1; then
        # Compilation succeeded, try to run
        if "$ESHKOL_TEST_BIN" > "$ESHKOL_TEST_OUT" 2>&1; then
            # Check for FAIL markers in output
            # A failure marker anywhere in the output fails the test — the old
            # `^FAIL`-anchored match never saw the indented `  <case>: FAIL`
            # form that most test programs actually print.
            if eshkol_test_output_has_failure "$ESHKOL_TEST_OUT"; then
                echo -e "${RED}ASSERTION FAIL${NC}"
                FAILED_TESTS+=("$test_name")
                ((FAIL++)) || true
                grep "FAIL:" "$ESHKOL_TEST_OUT" | head -5 | sed 's/^/    /'
            else
                echo -e "${GREEN}PASS${NC}"
                ((PASS++)) || true
            fi
        else
            exit_code=$?
            echo -e "${RED}RUNTIME FAIL (exit $exit_code)${NC}"
            FAILED_TESTS+=("$test_name")
            ((FAIL++)) || true
        fi
    else
        echo -e "${RED}COMPILE FAIL${NC}"
        FAILED_TESTS+=("$test_name")
        ((FAIL++)) || true
        tail -3 "$ESHKOL_TEST_COMPILE_LOG" 2>/dev/null | sed 's/^/    /'
    fi
done

echo ""

# ===== Summary =====
echo "========================================="
echo "  Signal Processing Test Results Summary"
echo "========================================="
TOTAL=$((PASS + FAIL))
echo "Total Tests:    $TOTAL"
echo -e "${GREEN}Passed:         $PASS${NC}"
echo -e "${RED}Failed:         $FAIL${NC}"
echo ""

if [ $FAIL -gt 0 ]; then
    echo "Failed Tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo ""
fi

# Clean up
rm -f "$ESHKOL_TEST_BIN" "$ESHKOL_TEST_COMPILE_LOG" "$ESHKOL_TEST_OUT"

# Exit with appropriate code
if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All signal processing tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some signal processing tests failed.${NC}"
    exit 1
fi
