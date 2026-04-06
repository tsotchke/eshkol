#!/bin/bash

# Eshkol GPU Test Suite
# Runs all GPU and softfloat tests

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
echo "  Eshkol GPU Test Suite"
echo "========================================="
echo ""

# Determine which build directory to use
# Override with: BUILD_DIR=build-cuda ./scripts/run_gpu_tests.sh
BUILD_DIR="${BUILD_DIR:-build}"

# Ensure build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: build directory '$BUILD_DIR' not found. Run cmake first.${NC}"
    exit 1
fi

# Check if compiler exists
if [ ! -f "$BUILD_DIR/eshkol-run" ]; then
    echo -e "${RED}Error: eshkol-run not found in '$BUILD_DIR'. Run make first.${NC}"
    exit 1
fi

echo -e "${GREEN}Using build directory: $BUILD_DIR${NC}"
echo ""
echo "Testing all files in tests/gpu/ directory..."
echo ""

# Run each test
for test_file in tests/gpu/*.esk; do
    test_name=$(basename "$test_file")
    printf "Testing %-50s " "$test_name"

    # Clean up stale temp files before each test
    rm -f a.out a.out.tmp.o

    # Try to compile
    if ./"$BUILD_DIR"/eshkol-run "$test_file" -L./"$BUILD_DIR" > /dev/null 2>&1; then
        # Compilation succeeded, try to run
        if ./a.out > /tmp/gpu_test_output.txt 2>&1; then
            # Check for FAIL markers in output
            if grep -qE "^FAIL:|Failed:[[:space:]]+[1-9]" /tmp/gpu_test_output.txt; then
                echo -e "${YELLOW}FAIL MARKER${NC}"
                RUNTIME_ERRORS+=("$test_name")
                ((FAIL++)) || true
            else
                echo -e "${GREEN}PASS${NC}"
                ((PASS++)) || true
            fi
        else
            echo -e "${RED}RUNTIME FAIL${NC}"
            FAILED_TESTS+=("$test_name")
            ((FAIL++)) || true
        fi
    else
        echo -e "${RED}COMPILE FAIL${NC}"
        FAILED_TESTS+=("$test_name")
        ((COMPILE_FAIL++)) || true
        ((FAIL++)) || true
    fi
done

echo ""
echo "========================================="
echo "  Test Results Summary"
echo "========================================="
TOTAL=$(( PASS + FAIL ))
echo -e "Total Tests:        $TOTAL"
echo -e "${GREEN}Passed:             $PASS${NC}"
echo -e "${RED}Failed:             $FAIL${NC}"
echo -e "  Compile Failures: $COMPILE_FAIL"
echo -e "  Runtime Errors:   ${#RUNTIME_ERRORS[@]}"
echo ""

if [ $FAIL -gt 0 ]; then
    if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
        echo "Failed Tests:"
        for test in "${FAILED_TESTS[@]}"; do
            echo "  - $test"
        done
        echo ""
    fi

    if [ ${#RUNTIME_ERRORS[@]} -gt 0 ]; then
        echo "Tests with FAIL markers:"
        for test in "${RUNTIME_ERRORS[@]}"; do
            echo "  - $test"
        done
        echo ""
    fi
fi

if [ $TOTAL -gt 0 ]; then
    PASS_RATE=$(( PASS * 100 / TOTAL ))
    echo "Pass Rate: ${PASS_RATE}%"
fi

echo ""

# Clean up
rm -f /tmp/gpu_test_output.txt a.out a.out.tmp.o

# Exit with appropriate code
if [ $FAIL -eq 0 ]; then
    exit 0
else
    exit 1
fi
