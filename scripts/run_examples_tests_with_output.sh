#!/bin/bash

# Eshkol Examples Test Suite (Verbose Output)
# Same as run_examples_tests.sh but shows compile/runtime output for debugging

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "========================================="
echo "  Eshkol Examples Test Suite (Verbose)"
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

# Check if stdlib exists
if [ ! -f "build/stdlib.o" ]; then
    echo -e "${YELLOW}Warning: stdlib.o not found. Building...${NC}"
    cmake --build build --target stdlib
fi

# Allow specifying specific file or pattern as argument
if [ $# -gt 0 ]; then
    FILES="$@"
else
    FILES="examples/*.esk"
fi

echo "Testing: $FILES"
echo ""

PASS=0
FAIL=0

for test_file in $FILES; do
    if [ ! -f "$test_file" ]; then
        echo -e "${RED}File not found: $test_file${NC}"
        continue
    fi

    test_name=$(basename "$test_file")

    echo "========================================="
    echo -e "${CYAN}Testing: $test_name${NC}"
    echo "========================================="

    # Clean up stale temp files
    rm -f a.out a.out.tmp.o

    echo -e "${BLUE}[Compiling...]${NC}"

    # Try to compile (show output)
    if ./build/eshkol-run "$test_file" -L./build 2>&1; then
        echo ""
        echo -e "${BLUE}[Running...]${NC}"

        # Try to run (show output)
        if ./a.out 2>&1; then
            echo ""
            echo -e "${GREEN}✅ PASS${NC}"
            ((PASS++))
        else
            exit_code=$?
            echo ""
            if [ $exit_code -eq 139 ]; then
                echo -e "${RED}❌ SEGFAULT${NC}"
            else
                echo -e "${RED}❌ RUNTIME FAIL (exit $exit_code)${NC}"
            fi
            ((FAIL++))
        fi
    else
        echo ""
        echo -e "${RED}❌ COMPILE FAIL${NC}"
        ((FAIL++))
    fi

    echo ""
done

echo "========================================="
echo "  Summary"
echo "========================================="
echo -e "${GREEN}Passed: $PASS${NC}"
echo -e "${RED}Failed: $FAIL${NC}"

# Clean up
rm -f a.out a.out.tmp.o

if [ $FAIL -eq 0 ]; then
    exit 0
else
    exit 1
fi
