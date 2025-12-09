#!/bin/bash

# Eshkol Examples Test Suite
# Tests all examples and categorizes them by status
# Helps identify which examples to keep, modify, or remove

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Counters
PASS=0
COMPILE_FAIL=0
RUNTIME_FAIL=0
RUNTIME_ERROR=0

# Results arrays
declare -a WORKING_EXAMPLES
declare -a COMPILE_FAILURES
declare -a RUNTIME_FAILURES
declare -a RUNTIME_ERRORS

# Output directory for logs
LOG_DIR="/tmp/eshkol_examples_test"
mkdir -p "$LOG_DIR"

echo "========================================="
echo "  Eshkol Examples Test Suite"
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

echo "Testing all .esk files in examples/ directory..."
echo "Log files will be saved to: $LOG_DIR"
echo ""

# Count total examples
TOTAL=$(ls -1 examples/*.esk 2>/dev/null | wc -l | tr -d ' ')
CURRENT=0

# Run each test
for test_file in examples/*.esk; do
    ((CURRENT++))
    test_name=$(basename "$test_file")
    printf "[%3d/%3d] %-50s " "$CURRENT" "$TOTAL" "$test_name"

    # Clean up stale temp files before each test
    rm -f a.out a.out.tmp.o

    # Log file for this example
    compile_log="$LOG_DIR/${test_name%.esk}_compile.log"
    run_log="$LOG_DIR/${test_name%.esk}_run.log"

    # Try to compile
    if ./build/eshkol-run "$test_file" -L./build > "$compile_log" 2>&1; then
        # Compilation succeeded, try to run
        if ./a.out > "$run_log" 2>&1; then
            exit_code=$?
            # Check if there were any errors in output
            if grep -qi "error\|segmentation fault\|abort" "$run_log"; then
                echo -e "${YELLOW}⚠ RUNTIME ERROR${NC}"
                RUNTIME_ERRORS+=("$test_name")
                ((RUNTIME_ERROR++))
            else
                echo -e "${GREEN}✅ PASS${NC}"
                WORKING_EXAMPLES+=("$test_name")
                ((PASS++))
            fi
        else
            exit_code=$?
            if [ $exit_code -eq 139 ]; then
                echo -e "${RED}❌ SEGFAULT${NC}"
                RUNTIME_FAILURES+=("$test_name (segfault)")
            else
                echo -e "${RED}❌ RUNTIME FAIL (exit $exit_code)${NC}"
                RUNTIME_FAILURES+=("$test_name (exit $exit_code)")
            fi
            ((RUNTIME_FAIL++))
        fi
    else
        echo -e "${RED}❌ COMPILE FAIL${NC}"
        COMPILE_FAILURES+=("$test_name")
        ((COMPILE_FAIL++))
    fi
done

echo ""
echo "========================================="
echo "  Test Results Summary"
echo "========================================="
TOTAL_TESTS=$(( PASS + COMPILE_FAIL + RUNTIME_FAIL + RUNTIME_ERROR ))
echo -e "Total Examples:     $TOTAL_TESTS"
echo -e "${GREEN}Working:            $PASS${NC}"
echo -e "${RED}Compile Failures:   $COMPILE_FAIL${NC}"
echo -e "${RED}Runtime Failures:   $RUNTIME_FAIL${NC}"
echo -e "${YELLOW}Runtime Errors:     $RUNTIME_ERROR${NC}"
echo ""

# Calculate pass rate
if [ $TOTAL_TESTS -gt 0 ]; then
    PASS_RATE=$(( PASS * 100 / TOTAL_TESTS ))
    echo "Pass Rate: ${PASS_RATE}%"
    echo ""
fi

# Report working examples
if [ ${#WORKING_EXAMPLES[@]} -gt 0 ]; then
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}  Working Examples (KEEP)${NC}"
    echo -e "${GREEN}=========================================${NC}"
    for example in "${WORKING_EXAMPLES[@]}"; do
        echo "  ✅ $example"
    done
    echo ""
fi

# Report compile failures
if [ ${#COMPILE_FAILURES[@]} -gt 0 ]; then
    echo -e "${RED}=========================================${NC}"
    echo -e "${RED}  Compile Failures (REVIEW/FIX)${NC}"
    echo -e "${RED}=========================================${NC}"
    for example in "${COMPILE_FAILURES[@]}"; do
        echo "  ❌ $example"
        # Show first few lines of error
        log_file="$LOG_DIR/${example%.esk}_compile.log"
        if [ -f "$log_file" ]; then
            echo "     Error: $(grep -m1 'error:' "$log_file" 2>/dev/null || head -1 "$log_file")"
        fi
    done
    echo ""
fi

# Report runtime failures
if [ ${#RUNTIME_FAILURES[@]} -gt 0 ]; then
    echo -e "${RED}=========================================${NC}"
    echo -e "${RED}  Runtime Failures (REVIEW/FIX)${NC}"
    echo -e "${RED}=========================================${NC}"
    for example in "${RUNTIME_FAILURES[@]}"; do
        echo "  ❌ $example"
    done
    echo ""
fi

# Report runtime errors
if [ ${#RUNTIME_ERRORS[@]} -gt 0 ]; then
    echo -e "${YELLOW}=========================================${NC}"
    echo -e "${YELLOW}  Runtime Errors (REVIEW)${NC}"
    echo -e "${YELLOW}=========================================${NC}"
    for example in "${RUNTIME_ERRORS[@]}"; do
        echo "  ⚠ $example"
    done
    echo ""
fi

echo "========================================="
echo "  Log files saved to: $LOG_DIR"
echo "========================================="
echo ""
echo "To view compile errors:  cat $LOG_DIR/<example>_compile.log"
echo "To view runtime output:  cat $LOG_DIR/<example>_run.log"
echo ""

# Clean up
rm -f a.out a.out.tmp.o

# Exit with appropriate code
if [ $COMPILE_FAIL -eq 0 ] && [ $RUNTIME_FAIL -eq 0 ]; then
    exit 0
else
    exit 1
fi
