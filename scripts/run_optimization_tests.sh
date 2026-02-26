#!/bin/bash

# Eshkol Optimization Algorithms Test Suite
# Tests gradient descent, Adam, L-BFGS, conjugate gradient

set -e

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

echo "========================================="
echo "  Eshkol Optimization Algorithms Tests"
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
OPT_TEST_DIR="tests/ml"

if [ ! -d "$OPT_TEST_DIR" ]; then
    echo -e "${RED}Error: $OPT_TEST_DIR directory not found.${NC}"
    exit 1
fi

# Count test files
TEST_COUNT=$(find "$OPT_TEST_DIR" -name "*optimization*" 2>/dev/null | wc -l | tr -d ' ')
if [ "$TEST_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}No optimization test files found.${NC}"
    exit 0
fi

echo "Found $TEST_COUNT test file(s)"
echo ""

# Run optimization test files specifically
for test_file in "$OPT_TEST_DIR"/*optimization*.esk; do
    if [ ! -f "$test_file" ]; then
        continue
    fi

    test_name=$(basename "$test_file")
    printf "Testing %-45s " "$test_name"

    # Clean up stale artifacts
    rm -f a.out

    # Try to compile
    if ./$BUILD_DIR/eshkol-run "$test_file" -L./$BUILD_DIR > /tmp/opt_compile.log 2>&1; then
        # Compilation succeeded, try to run
        if run_with_timeout 60 ./a.out > /tmp/opt_test_output.txt 2>&1; then
            # Check for FAIL markers in output
            if grep -q "FAIL:" /tmp/opt_test_output.txt; then
                echo -e "${RED}ASSERTION FAIL${NC}"
                FAILED_TESTS+=("$test_name")
                ((FAIL++)) || true
                grep "FAIL:" /tmp/opt_test_output.txt | head -5 | sed 's/^/    /'
            else
                echo -e "${GREEN}PASS${NC}"
                ((PASS++)) || true
            fi
        else
            exit_code=$?
            if [ $exit_code -eq 124 ]; then
                echo -e "${RED}TIMEOUT (60s)${NC}"
            else
                echo -e "${RED}RUNTIME FAIL (exit $exit_code)${NC}"
            fi
            FAILED_TESTS+=("$test_name")
            ((FAIL++)) || true
        fi
    else
        echo -e "${RED}COMPILE FAIL${NC}"
        FAILED_TESTS+=("$test_name")
        ((FAIL++)) || true
        tail -3 /tmp/opt_compile.log 2>/dev/null | sed 's/^/    /'
    fi
done

echo ""

# ===== Summary =====
echo "========================================="
echo "  Optimization Test Results Summary"
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
rm -f a.out /tmp/opt_compile.log /tmp/opt_test_output.txt

# Exit with appropriate code
if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All optimization tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some optimization tests failed.${NC}"
    exit 1
fi
