#!/bin/bash

# Eshkol HoTT Type Checker C++ Unit Tests
# Compiles and runs the C++ type system tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "  Eshkol HoTT Type Checker C++ Tests   "
echo "========================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Check if llvm-config exists
if ! command -v llvm-config &> /dev/null; then
    echo -e "${RED}Error: llvm-config not found. Please install LLVM.${NC}"
    exit 1
fi

# Get LLVM configuration
LLVM_CXXFLAGS=$(llvm-config --cxxflags)
LLVM_LDFLAGS=$(llvm-config --ldflags)
LLVM_LIBS=$(llvm-config --libs all)
LLVM_SYSTEM_LIBS=$(llvm-config --system-libs)

# Test files
HOTT_TYPES_TEST="$PROJECT_DIR/tests/types/hott_types_test.cpp"
TYPE_CHECKER_TEST="$PROJECT_DIR/tests/types/type_checker_test.cpp"

# Source files needed
SOURCES="$PROJECT_DIR/lib/types/hott_types.cpp $PROJECT_DIR/lib/types/type_checker.cpp $PROJECT_DIR/lib/types/dependent.cpp"

# Output directory
BUILD_DIR="$PROJECT_DIR/build"
mkdir -p "$BUILD_DIR"

# Compile and run each test
compile_and_run_test() {
    local test_file="$1"
    local test_name=$(basename "$test_file" .cpp)
    local output="$BUILD_DIR/$test_name"

    printf "Compiling %-40s " "$test_name..."

    # Compile the test
    if g++ -std=c++20 \
        -I"$PROJECT_DIR/inc" \
        $LLVM_CXXFLAGS \
        $SOURCES \
        "$test_file" \
        $LLVM_LDFLAGS $LLVM_LIBS $LLVM_SYSTEM_LIBS \
        -o "$output" 2>&1; then

        echo -e "${GREEN}OK${NC}"

        printf "Running   %-40s " "$test_name..."
        echo ""

        # Run the test
        if "$output"; then
            echo -e "${GREEN}PASSED${NC}"
            return 0
        else
            echo -e "${RED}FAILED${NC}"
            return 1
        fi
    else
        echo -e "${RED}COMPILE FAILED${NC}"
        return 1
    fi
}

# Track results
TOTAL=0
PASSED=0
FAILED=0

# Run HoTT types test
if [ -f "$HOTT_TYPES_TEST" ]; then
    echo ""
    echo "--- HoTT Types Test ---"
    ((TOTAL++)) || true
    if compile_and_run_test "$HOTT_TYPES_TEST"; then
        ((PASSED++)) || true
    else
        ((FAILED++)) || true
    fi
fi

# Run Type Checker test
if [ -f "$TYPE_CHECKER_TEST" ]; then
    echo ""
    echo "--- Type Checker Test ---"
    ((TOTAL++)) || true
    if compile_and_run_test "$TYPE_CHECKER_TEST"; then
        ((PASSED++)) || true
    else
        ((FAILED++)) || true
    fi
fi

# Summary
echo ""
echo "========================================="
echo "  C++ Test Results Summary"
echo "========================================="
echo -e "Total:  $TOTAL"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}ALL C++ TYPE TESTS PASSED!${NC}"
    exit 0
else
    echo -e "${RED}SOME C++ TYPE TESTS FAILED!${NC}"
    exit 1
fi
