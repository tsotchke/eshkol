#!/bin/bash

# Eshkol Control Flow Test Suite
# Runs all control flow tests and reports results

set -e

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
    rm -f a.out a.out.tmp.o

    # Compile and run the test
    if ./build/eshkol-run "$test_file" -L./build > /dev/null 2>&1; then
        if ./a.out > /tmp/test_output.txt 2>&1; then
            # Check for failures in output
            if grep -q "FAIL:" /tmp/test_output.txt; then
                echo -e "${RED}❌ TESTS FAILED${NC}"
                grep "FAIL:" /tmp/test_output.txt
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
rm -f /tmp/test_output.txt a.out

# Exit with appropriate code
if [ $FAIL -eq 0 ]; then
    exit 0
else
    exit 1
fi
