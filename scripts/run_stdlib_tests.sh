#!/bin/bash

# Eshkol Stdlib Test Suite
# Runs all stdlib tests and reports results

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
echo "  Eshkol Stdlib Test Suite"
echo "========================================="
echo ""

# Ensure build directory exists
if [ ! -d "build" ]; then
    echo -e "${RED}Error: build directory not found. Run cmake first.${NC}"
    exit 1
fi

# Check if compiler exists
if [ ! -f "build/eshkol-run" ]; then
    echo -e "${RED}Error: eshkol-run not found. Run make first.${NC}"
    exit 1
fi

echo "Testing all files in tests/stdlib/ directory..."
echo ""

# Run each test
for test_file in tests/stdlib/*.esk; do
    test_name=$(basename "$test_file")
    printf "Testing %-50s " "$test_name"

    # Clean up stale temp files before each test
    rm -f a.out a.out.tmp.o

    # Try to compile
    if ./build/eshkol-run "$test_file" -L./build > /dev/null 2>&1; then
        # Compilation succeeded, try to run
        if ./a.out > /tmp/test_output.txt 2>&1; then
            # Check if there were any errors in output
            if grep -q "error:" /tmp/test_output.txt; then
                echo -e "${YELLOW}⚠ RUNTIME ERROR${NC}"
                RUNTIME_ERRORS+=("$test_name")
                ((FAIL++)) || true
            else
                echo -e "${GREEN}✅ PASS${NC}"
                ((PASS++)) || true
            fi
        else
            echo -e "${RED}❌ RUNTIME FAIL${NC}"
            FAILED_TESTS+=("$test_name")
            ((FAIL++)) || true
        fi
    else
        echo -e "${RED}❌ COMPILE FAIL${NC}"
        FAILED_TESTS+=("$test_name")
        ((COMPILE_FAIL++)) || true
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
echo -e "  Compile Failures: $COMPILE_FAIL"
echo -e "  Runtime Errors:   ${#RUNTIME_ERRORS[@]}"
echo ""

if [ $FAIL -gt 0 ]; then
    echo "Failed Tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo ""

    if [ ${#RUNTIME_ERRORS[@]} -gt 0 ]; then
        echo "Runtime Errors:"
        for test in "${RUNTIME_ERRORS[@]}"; do
            echo "  - $test"
        done
        echo ""
    fi
fi

# Calculate pass rate
TOTAL=$(( PASS + FAIL ))
if [ $TOTAL -gt 0 ]; then
    PASS_RATE=$(( PASS * 100 / TOTAL ))
    echo "Pass Rate: ${PASS_RATE}%"
fi

echo ""

# Clean up
rm -f /tmp/test_output.txt a.out

# Exit with appropriate code
if [ $FAIL -eq 0 ]; then
    exit 0
else
    exit 1
fi
