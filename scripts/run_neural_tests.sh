#!/bin/bash

# Eshkol Neural Network Test Suite Runner
# Tests all neural network functionality

set -e  # Exit on first error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
TOTAL=0
PASSED=0
FAILED=0
COMPILE_FAILED=0
RUNTIME_FAILED=0

# Failed test tracking
declare -a FAILED_TESTS

echo "========================================="
echo "  Eshkol Neural Network Test Suite"
echo "========================================="
echo ""
echo "Testing all neural network tests..."
echo ""

# Test directory
NEURAL_DIR="tests/neural"

# Function to run a single test
run_test() {
    local test_file=$1
    local test_name=$(basename "$test_file")

    TOTAL=$((TOTAL + 1))

    printf "Testing %-50s" "$test_name"

    # Compile test
    if ! ./build/eshkol-run "$test_file" > /tmp/compile_output.txt 2>&1; then
        echo -e "${RED}❌ COMPILE FAIL${NC}"
        COMPILE_FAILED=$((COMPILE_FAILED + 1))
        FAILED=$((FAILED + 1))
        FAILED_TESTS+=("$test_name")
        return
    fi

    # Run compiled test
    if ! ./a.out > /tmp/test_output.txt 2>&1; then
        echo -e "${RED}❌ RUNTIME FAIL${NC}"
        RUNTIME_FAILED=$((RUNTIME_FAILED + 1))
        FAILED=$((FAILED + 1))
        FAILED_TESTS+=("$test_name")
        return
    fi

    echo -e "${GREEN}✅ PASS${NC}"
    PASSED=$((PASSED + 1))
}

# Run all tests in neural directory
if [ -d "$NEURAL_DIR" ]; then
    for test_file in "$NEURAL_DIR"/*.esk; do
        if [ -f "$test_file" ]; then
            run_test "$test_file"
        fi
    done
else
    echo -e "${RED}Neural test directory not found: $NEURAL_DIR${NC}"
    exit 1
fi

echo ""
echo "========================================="
echo "  Neural Network Test Results Summary"
echo "========================================="
echo "Total Tests:    $TOTAL"
echo "Passed:         $PASSED"
echo "Failed:         $FAILED"
echo "  Compile Failures: $COMPILE_FAILED"
echo "  Runtime Errors:   $RUNTIME_FAILED"
echo ""

if [ $FAILED -gt 0 ]; then
    echo "Failed Tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo ""
fi

# Calculate pass rate
if [ $TOTAL -gt 0 ]; then
    PASS_RATE=$((100 * PASSED / TOTAL))
    echo "Pass Rate: ${PASS_RATE}%"
    echo ""
fi

# Clean up
rm -f a.out

# Exit with error code if any tests failed
if [ $FAILED -gt 0 ]; then
    exit 1
fi

exit 0
