#!/bin/bash

# JSON Test Suite with Full Output Capture
# Shows complete compilation and runtime output for all tests

set +e  # Don't exit on error, we want to see all failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASS=0
FAIL=0
COMPILE_FAIL=0
RUNTIME_FAIL=0

# Output directory
OUTPUT_DIR="json_test_outputs"
mkdir -p "$OUTPUT_DIR"

# Results file
RESULTS_FILE="$OUTPUT_DIR/json_results_summary.txt"
> "$RESULTS_FILE"  # Clear file

# Results arrays
declare -a FAILED_TESTS
declare -a SEGFAULT_TESTS

echo "========================================="
echo "  JSON Test Suite - Verbose Mode"
echo "========================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# Test directory
JSON_DIR="tests/json"

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

# Function to run a single test with full output
run_test_verbose() {
    local test_file=$1
    local test_name=$(basename "$test_file")
    local output_file="$OUTPUT_DIR/${test_name%.esk}_full_output.txt"

    echo "========================================"
    echo "Testing: $test_name"
    echo "========================================"

    # Clean up stale temp files before each test
    rm -f a.out a.out.tmp.o

    # Clear output file
    > "$output_file"

    # Add header
    echo "========================================" >> "$output_file"
    echo "Test: $test_name" >> "$output_file"
    echo "File: $test_file" >> "$output_file"
    echo "Time: $(date)" >> "$output_file"
    echo "========================================" >> "$output_file"
    echo "" >> "$output_file"

    # Try to compile with full output
    echo "COMPILATION OUTPUT:" >> "$output_file"
    echo "----------------------------------------" >> "$output_file"
    if ./build/eshkol-run "$test_file" -L./build >> "$output_file" 2>&1; then
        echo "----------------------------------------" >> "$output_file"
        echo "COMPILATION: SUCCESS" >> "$output_file"
        echo "" >> "$output_file"

        echo -e "${GREEN}  ✅ Compilation succeeded${NC}"

        # Try to run with full output
        echo "RUNTIME OUTPUT:" >> "$output_file"
        echo "----------------------------------------" >> "$output_file"
        if ./a.out >> "$output_file" 2>&1; then
            echo "----------------------------------------" >> "$output_file"
            echo "RUNTIME: SUCCESS" >> "$output_file"
            echo "" >> "$output_file"
            echo "FINAL STATUS: PASS" >> "$output_file"

            echo -e "${GREEN}  ✅ Runtime succeeded${NC}"
            echo -e "${GREEN}  ✅ OVERALL: PASS${NC}"
            ((PASS++))

            # Show output
            echo ""
            echo "Runtime output:"
            tail -20 "$output_file" | grep -v "^========" | grep -v "^RUNTIME" | grep -v "^COMPILATION"

        else
            EXIT_CODE=$?
            echo "----------------------------------------" >> "$output_file"
            echo "RUNTIME: FAILED (exit code: $EXIT_CODE)" >> "$output_file"

            if [ $EXIT_CODE -eq 139 ] || [ $EXIT_CODE -eq 134 ]; then
                echo "ERROR TYPE: SEGMENTATION FAULT" >> "$output_file"
                echo -e "${RED}  ❌ Runtime SEGFAULT${NC}"
                SEGFAULT_TESTS+=("$test_name")
            elif [ $EXIT_CODE -eq 124 ]; then
                echo "ERROR TYPE: TIMEOUT (>5s)" >> "$output_file"
                echo -e "${RED}  ❌ Runtime TIMEOUT${NC}"
            else
                echo "ERROR TYPE: CRASH/ERROR" >> "$output_file"
                echo -e "${RED}  ❌ Runtime error${NC}"
            fi

            echo "FINAL STATUS: RUNTIME FAIL" >> "$output_file"
            echo -e "${RED}  ❌ OVERALL: FAIL${NC}"
            ((RUNTIME_FAIL++))
            ((FAIL++))
            FAILED_TESTS+=("$test_name")

            # Show last lines of output
            echo ""
            echo "Runtime output (last 30 lines):"
            tail -30 "$output_file" | grep -v "^========" | grep -v "^COMPILATION"
        fi
    else
        echo "----------------------------------------" >> "$output_file"
        echo "COMPILATION: FAILED" >> "$output_file"
        echo "FINAL STATUS: COMPILE FAIL" >> "$output_file"

        echo -e "${RED}  ❌ Compilation failed${NC}"
        echo -e "${RED}  ❌ OVERALL: FAIL${NC}"
        ((COMPILE_FAIL++))
        ((FAIL++))
        FAILED_TESTS+=("$test_name")

        # Show compilation errors
        echo ""
        echo "Compilation errors (last 30 lines):"
        tail -30 "$output_file"
    fi

    echo ""
    echo "Full output saved to: $output_file"
    echo ""
}

# Run all tests in json directory
if [ -d "$JSON_DIR" ]; then
    echo "Running tests in $JSON_DIR..."
    echo ""
    for test_file in "$JSON_DIR"/*.esk; do
        if [ -f "$test_file" ]; then
            run_test_verbose "$test_file"
        fi
    done
else
    echo -e "${RED}Error: $JSON_DIR directory not found${NC}"
    exit 1
fi

# Create summary
echo "=========================================" | tee -a "$RESULTS_FILE"
echo "  JSON Test Results Summary" | tee -a "$RESULTS_FILE"
echo "=========================================" | tee -a "$RESULTS_FILE"
TOTAL=$((PASS + FAIL))
echo "Total Tests:        $TOTAL" | tee -a "$RESULTS_FILE"
echo -e "${GREEN}Passed:             $PASS${NC}" | tee -a "$RESULTS_FILE"
echo -e "${RED}Failed:             $FAIL${NC}" | tee -a "$RESULTS_FILE"
echo "  Compile Failures: $COMPILE_FAIL" | tee -a "$RESULTS_FILE"
echo "  Runtime Failures: $RUNTIME_FAIL" | tee -a "$RESULTS_FILE"
echo "  Segfaults:        ${#SEGFAULT_TESTS[@]}" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

if [ $FAIL -gt 0 ]; then
    echo "Failed Tests:" | tee -a "$RESULTS_FILE"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test" | tee -a "$RESULTS_FILE"
    done
    echo "" | tee -a "$RESULTS_FILE"
fi

if [ ${#SEGFAULT_TESTS[@]} -gt 0 ]; then
    echo "Segfault Tests:" | tee -a "$RESULTS_FILE"
    for test in "${SEGFAULT_TESTS[@]}"; do
        echo "  - $test" | tee -a "$RESULTS_FILE"
    done
    echo "" | tee -a "$RESULTS_FILE"
fi

# Calculate pass rate
if [ $TOTAL -gt 0 ]; then
    PASS_RATE=$((PASS * 100 / TOTAL))
    echo "Pass Rate: ${PASS_RATE}%" | tee -a "$RESULTS_FILE"
fi

echo "" | tee -a "$RESULTS_FILE"
echo -e "${BLUE}All test outputs saved in: $OUTPUT_DIR${NC}"
echo -e "${BLUE}Summary file: $RESULTS_FILE${NC}"
echo ""

# Clean up
rm -f a.out a.out.tmp.o

# Exit with appropriate code
if [ $FAIL -gt 0 ]; then
    exit 1
else
    exit 0
fi
