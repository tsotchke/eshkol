#!/bin/bash

# System Test Suite (Hash Tables, File I/O, etc.)
# Runs all system-level tests

set +e  # Don't exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# Counters
PASS=0
FAIL=0

echo "========================================="
echo "  System Test Suite"
echo "========================================="
echo ""

TEST_DIR="tests/system"

if [ ! -d "$TEST_DIR" ]; then
    echo -e "${RED}Test directory not found: $TEST_DIR${NC}"
    exit 1
fi

for test_file in "$TEST_DIR"/*.esk; do
    if [ ! -f "$test_file" ]; then
        continue
    fi

    test_name=$(basename "$test_file")
    printf "Testing: %-40s " "$test_name"

    # Compile
    if ! ./build/eshkol-run "$test_file" > /dev/null 2>&1; then
        echo -e "${RED}COMPILE FAIL${NC}"
        ((FAIL++))
        continue
    fi

    # Run
    if ./a.out > /dev/null 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        ((PASS++))
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 139 ] || [ $EXIT_CODE -eq 134 ]; then
            echo -e "${RED}SEGFAULT${NC}"
        else
            echo -e "${RED}RUNTIME FAIL (exit $EXIT_CODE)${NC}"
        fi
        ((FAIL++))
    fi
done

# Summary
echo ""
echo "========================================="
TOTAL=$((PASS + FAIL))
echo "Total: $TOTAL  Passed: $PASS  Failed: $FAIL"
if [ $TOTAL -gt 0 ]; then
    PASS_RATE=$((PASS * 100 / TOTAL))
    echo "Pass Rate: ${PASS_RATE}%"
fi
echo "========================================="

rm -f a.out

if [ $FAIL -gt 0 ]; then
    exit 1
else
    exit 0
fi
