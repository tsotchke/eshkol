#!/bin/bash

# Eshkol TCO (Tail Call Optimization) Test Suite
# Tests that TCO works correctly with nested letrec expressions.
# These patterns previously caused stack overflow due to TCO context corruption.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Portable timeout wrapper (macOS has no timeout command)
run_with_timeout() {
    local timeout_secs=$1
    shift
    if command -v timeout &>/dev/null; then
        timeout "$timeout_secs" "$@"
    elif command -v gtimeout &>/dev/null; then
        gtimeout "$timeout_secs" "$@"
    else
        "$@" &
        local pid=$!
        ( sleep "$timeout_secs"; kill -9 $pid 2>/dev/null ) &
        local watcher=$!
        wait $pid 2>/dev/null
        local exit_code=$?
        kill $watcher 2>/dev/null 2>&1
        wait $watcher 2>/dev/null 2>&1
        return $exit_code
    fi
}

# Counters
PASS=0
FAIL=0

# Results arrays
declare -a FAILED_TESTS

echo "========================================="
echo "  Eshkol TCO (Tail Call Optimization) Tests"
echo "========================================="
echo ""

# Check for build directory
BUILD_DIR="build"

if [ ! -d "$BUILD_DIR" ] || [ ! -f "$BUILD_DIR/eshkol-run" ]; then
    echo -e "${RED}Error: Build directory not found or eshkol-run missing.${NC}"
    echo "Please build first: cd build && cmake .. && make -j8"
    exit 1
fi

echo -e "${GREEN}Using build directory: $BUILD_DIR${NC}"
echo ""

# Check for test directory
TCO_TEST_DIR="tests/tco"

if [ ! -d "$TCO_TEST_DIR" ]; then
    echo -e "${RED}Error: $TCO_TEST_DIR directory not found.${NC}"
    exit 1
fi

# Count test files
TEST_COUNT=$(ls "$TCO_TEST_DIR"/*.esk 2>/dev/null | wc -l | tr -d ' ')
if [ "$TEST_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}No TCO test files found (*.esk).${NC}"
    exit 0
fi

echo "Found $TEST_COUNT test file(s)"
echo ""

# Run each test file
for test_file in "$TCO_TEST_DIR"/*.esk; do
    test_name=$(basename "$test_file" .esk)
    echo -n "  $test_name ... "

    # Compile
    TEMP_BIN=$(mktemp /tmp/tco_test_XXXXXX)
    set +e
    COMPILE_OUTPUT=$(./$BUILD_DIR/eshkol-run "$test_file" -o "$TEMP_BIN" 2>&1)
    COMPILE_EXIT=$?
    set -e

    if [ $COMPILE_EXIT -ne 0 ]; then
        echo -e "${RED}COMPILE FAIL${NC}"
        echo "    Compile output: $COMPILE_OUTPUT"
        ((FAIL++)) || true
        FAILED_TESTS+=("$test_name (compile)")
        rm -f "$TEMP_BIN"
        continue
    fi

    # Run with timeout (TCO bugs cause infinite recursion → stack overflow)
    RUN_OUTPUT=$(run_with_timeout 60 "$TEMP_BIN" 2>&1)
    RUN_EXIT=$?

    rm -f "$TEMP_BIN"

    if [ $RUN_EXIT -ne 0 ]; then
        echo -e "${RED}RUNTIME FAIL (exit $RUN_EXIT)${NC}"
        echo "    Output: $RUN_OUTPUT"
        ((FAIL++)) || true
        FAILED_TESTS+=("$test_name (runtime exit $RUN_EXIT)")
        continue
    fi

    # Check for FAIL in output
    if echo "$RUN_OUTPUT" | grep -q "FAIL"; then
        echo -e "${RED}FAIL${NC}"
        echo "    Output: $RUN_OUTPUT"
        ((FAIL++)) || true
        FAILED_TESTS+=("$test_name (wrong result)")
        continue
    fi

    # Check for PASS in output
    if echo "$RUN_OUTPUT" | grep -q "PASS"; then
        echo -e "${GREEN}PASS${NC}"
        ((PASS++)) || true
    else
        echo -e "${YELLOW}UNKNOWN${NC}"
        echo "    Output: $RUN_OUTPUT"
        ((FAIL++)) || true
        FAILED_TESTS+=("$test_name (no PASS/FAIL)")
    fi
done

echo ""
echo "========================================="
echo "  TCO Test Results"
echo "========================================="
echo "Passed: $PASS"
echo "Failed: $FAIL"

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed tests:${NC}"
    for t in "${FAILED_TESTS[@]}"; do
        echo "  - $t"
    done
fi

echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All TCO tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some TCO tests failed.${NC}"
    exit 1
fi
