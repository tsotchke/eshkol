#!/bin/bash

# Eshkol Bignum Test Suite
# Tests arbitrary-precision integer operations

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

echo "========================================="
echo "  Eshkol Bignum Tests"
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
BIGNUM_TEST_DIR="tests/bignum"

if [ ! -d "$BIGNUM_TEST_DIR" ]; then
    echo -e "${RED}Error: $BIGNUM_TEST_DIR directory not found.${NC}"
    exit 1
fi

# Count test files
TEST_COUNT=$(find "$BIGNUM_TEST_DIR" -name "*.esk" 2>/dev/null | wc -l | tr -d ' ')
if [ "$TEST_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}No bignum test files found (*.esk).${NC}"
    exit 0
fi

echo "Found $TEST_COUNT test file(s)"
echo ""

# Run each .esk test
for test_file in "$BIGNUM_TEST_DIR"/*.esk; do
    if [ ! -f "$test_file" ]; then
        continue
    fi

    test_name=$(basename "$test_file")
    printf "Testing %-45s " "$test_name"

    # Clean up stale artifacts
    rm -f a.out

    # Try to compile
    if ./$BUILD_DIR/eshkol-run "$test_file" -L./$BUILD_DIR > /tmp/bignum_compile.log 2>&1; then
        # Compilation succeeded, try to run
        if ./a.out > /tmp/bignum_test_output.txt 2>&1; then
            # Check for FAIL markers in output
            if grep -q "^FAIL:" /tmp/bignum_test_output.txt; then
                echo -e "${RED}ASSERTION FAIL${NC}"
                FAILED_TESTS+=("$test_name")
                ((FAIL++)) || true
                grep "^FAIL:" /tmp/bignum_test_output.txt | head -5 | sed 's/^/    /'
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
        tail -3 /tmp/bignum_compile.log 2>/dev/null | sed 's/^/    /'
    fi
done

echo ""

# ===== Summary =====
echo "========================================="
echo "  Bignum Test Results Summary"
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
rm -f a.out /tmp/bignum_compile.log /tmp/bignum_test_output.txt

# Exit with appropriate code
if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All bignum tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some bignum tests failed.${NC}"
    exit 1
fi
