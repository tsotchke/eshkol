#!/bin/bash

# Eshkol REPL Test Suite
# Runs all REPL tests and reports results

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASS=0
FAIL=0

# Results array
declare -a FAILED_TESTS

echo "========================================="
echo "  Eshkol REPL Test Suite"
echo "========================================="
echo ""

# Ensure build directory exists
if [ ! -d "build" ]; then
    echo -e "${RED}Error: build directory not found. Run cmake first.${NC}"
    exit 1
fi

# Check if REPL exists
if [ ! -f "build/eshkol-repl" ]; then
    echo -e "${RED}Error: eshkol-repl not found. Run make first.${NC}"
    exit 1
fi

# Check if test directory exists
if [ ! -d "tests/repl" ]; then
    echo -e "${YELLOW}Warning: tests/repl directory not found. Creating...${NC}"
    mkdir -p tests/repl
fi

echo "Testing all files in tests/repl/ directory..."
echo ""

# Run each test
for test_file in tests/repl/*.esk; do
    # Skip if no files found
    [ -e "$test_file" ] || continue

    test_name=$(basename "$test_file")
    printf "Testing %-50s " "$test_name"

    # Run the test through REPL (add :quit at the end)
    # Use timeout command if available, otherwise just run directly
    if command -v timeout > /dev/null 2>&1; then
        # Linux has timeout
        { cat "$test_file"; echo ""; echo ":quit"; } | timeout 10 ./build/eshkol-repl > /tmp/repl_test_output.txt 2>&1 || true
        EXIT_CODE=$?
    else
        # macOS - run directly (no timeout needed, tests are fast)
        { cat "$test_file"; echo ""; echo ":quit"; } | ./build/eshkol-repl > /tmp/repl_test_output.txt 2>&1 || true
        EXIT_CODE=$?
    fi

    # Check for errors in output
    if [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 142 ]; then
        echo -e "${YELLOW}⚠ TIMEOUT${NC}"
        FAILED_TESTS+=("$test_name (timeout)")
        ((FAIL++)) || true
    elif grep -q "error:" /tmp/repl_test_output.txt 2>/dev/null; then
        echo -e "${RED}❌ RUNTIME ERROR${NC}"
        FAILED_TESTS+=("$test_name")
        ((FAIL++)) || true
    elif grep -q "Segmentation fault" /tmp/repl_test_output.txt 2>/dev/null; then
        echo -e "${RED}❌ SEGFAULT${NC}"
        FAILED_TESTS+=("$test_name")
        ((FAIL++)) || true
    else
        echo -e "${GREEN}✅ PASS${NC}"
        ((PASS++)) || true
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

if [ $FAIL -gt 0 ]; then
    echo "Failed Tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo ""
fi

# Calculate pass rate
TOTAL=$(( PASS + FAIL ))
if [ $TOTAL -gt 0 ]; then
    PASS_RATE=$(( PASS * 100 / TOTAL ))
    echo "Pass Rate: ${PASS_RATE}%"
fi

echo ""

# Clean up
rm -f /tmp/repl_test_output.txt

# Exit with appropriate code
if [ $FAIL -eq 0 ]; then
    exit 0
else
    exit 1
fi
