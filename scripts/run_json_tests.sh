#!/bin/bash

# JSON Test Suite Validation Script
# Runs all tests in tests/json/ directory

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASS=0
FAIL=0
COMPILE_FAIL=0

# Results array
declare -a FAILED_TESTS
declare -a RUNTIME_ERRORS

echo "========================================="
echo "  JSON Test Suite Validation"
echo "========================================="
echo ""

# Honour $BUILD_DIR (CI passes it via the matrix); fall back to "build" for plain local runs.
BUILD_DIR="${BUILD_DIR:-build}"
FAILURE_LINES="${ESHKOL_TEST_FAILURE_LINES:-40}"

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

echo "Testing all files in tests/json/ directory..."
echo ""

# Run each test
for test_file in tests/json/*.esk; do
    test_name=$(basename "$test_file")
    printf "Testing %-50s " "$test_name"

    # Clean up stale temp files before each test
    rm -f a.out a.out.tmp.o /tmp/test_output.txt /tmp/test_compile_output.txt

    # Try to compile
    if ./$BUILD_DIR/eshkol-run -L./$BUILD_DIR "$test_file" > /tmp/test_compile_output.txt 2>&1; then
        # Compilation succeeded, try to run
        if ./a.out > /tmp/test_output.txt 2>&1; then
            # Check if there were any errors in output
            if grep -q "error:" /tmp/test_output.txt; then
                echo -e "${YELLOW}RUNTIME ERROR${NC}"
                head -n "$FAILURE_LINES" /tmp/test_output.txt | sed 's/^/    /'
                RUNTIME_ERRORS+=("$test_name")
                ((FAIL++)) || true
            else
                echo -e "${GREEN}PASS${NC}"
                ((PASS++)) || true
            fi
        else
            echo -e "${RED}RUNTIME FAIL${NC}"
            head -n "$FAILURE_LINES" /tmp/test_output.txt | sed 's/^/    /'
            FAILED_TESTS+=("$test_name")
            ((FAIL++)) || true
        fi
    else
        echo -e "${RED}COMPILE FAIL${NC}"
        head -n "$FAILURE_LINES" /tmp/test_compile_output.txt | sed 's/^/    /'
        FAILED_TESTS+=("$test_name")
        ((COMPILE_FAIL++)) || true
        ((FAIL++)) || true
    fi
done

echo ""
echo "========================================="
echo "  Test Results Summary"
echo "========================================="
echo -e "Total: $((PASS + FAIL)) tests"
echo -e "${GREEN}Passed: $PASS${NC}"
echo -e "${RED}Failed: $FAIL${NC}"
if [ $COMPILE_FAIL -gt 0 ]; then
    echo -e "${RED}Compile failures: $COMPILE_FAIL${NC}"
fi

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo ""
    echo "Failed tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
fi

if [ ${#RUNTIME_ERRORS[@]} -gt 0 ]; then
    echo ""
    echo "Tests with runtime errors:"
    for test in "${RUNTIME_ERRORS[@]}"; do
        echo "  - $test"
    done
fi

# Clean up
rm -f a.out a.out.tmp.o /tmp/test_output.txt /tmp/test_compile_output.txt

echo ""
if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All JSON tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
