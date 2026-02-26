#!/bin/bash

# Eshkol REPL Test Suite - Verbose Mode
# Iterates tests/repl/*.esk and shows full REPL output for debugging

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASS=0
FAIL=0

# Output directory
OUTPUT_DIR="repl_test_outputs"
mkdir -p "$OUTPUT_DIR"

# Results file
RESULTS_FILE="$OUTPUT_DIR/repl_results_summary.txt"
> "$RESULTS_FILE"

# Results arrays
declare -a FAILED_TESTS
declare -a SEGFAULT_TESTS

echo "========================================="
echo "  Eshkol REPL Test Suite - Verbose Mode"
echo "========================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
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

# Test directory
TEST_DIR="tests/repl"

# Check if test directory exists
if [ ! -d "$TEST_DIR" ]; then
    echo -e "${RED}REPL test directory not found: $TEST_DIR${NC}"
    exit 1
fi

# Function to run a single REPL test with full output
run_test_verbose() {
    local test_file=$1
    local test_name=$(basename "$test_file")
    local output_file="$OUTPUT_DIR/${test_name%.esk}_full_output.txt"

    echo "========================================"
    echo "Testing: $test_name"
    echo "========================================"

    # Clear output file
    > "$output_file"

    # Add header
    echo "========================================" >> "$output_file"
    echo "Test: $test_name" >> "$output_file"
    echo "File: $test_file" >> "$output_file"
    echo "Time: $(date)" >> "$output_file"
    echo "========================================" >> "$output_file"
    echo "" >> "$output_file"

    # Show input
    echo ""
    echo "Input:"
    cat "$test_file" | sed 's/^/  /'
    echo ""

    # Run through REPL with :quit appended
    echo "REPL OUTPUT:" >> "$output_file"
    echo "----------------------------------------" >> "$output_file"
    { cat "$test_file"; echo ""; echo ":quit"; } | ./build/eshkol-repl >> "$output_file" 2>&1 || true
    EXIT_CODE=$?

    echo "----------------------------------------" >> "$output_file"

    # Show output
    echo "Output:"
    cat "$output_file" | grep -v "^========" | grep -v "^Test:" | grep -v "^File:" | grep -v "^Time:" | grep -v "^REPL OUTPUT:" | grep -v "^---" | sed 's/^/  /'
    echo ""

    # Check for failures
    if [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 142 ]; then
        echo "EXIT CODE: TIMEOUT ($EXIT_CODE)" >> "$output_file"
        echo "FINAL STATUS: TIMEOUT" >> "$output_file"
        echo -e "${YELLOW}  ⚠ TIMEOUT${NC}"
        FAILED_TESTS+=("$test_name (timeout)")
        ((FAIL++)) || true
    elif grep -q "Segmentation fault" "$output_file" 2>/dev/null; then
        echo "FINAL STATUS: SEGFAULT" >> "$output_file"
        echo -e "${RED}  ❌ SEGFAULT${NC}"
        SEGFAULT_TESTS+=("$test_name")
        FAILED_TESTS+=("$test_name")
        ((FAIL++)) || true
    elif grep -q "error:" "$output_file" 2>/dev/null; then
        echo "FINAL STATUS: RUNTIME ERROR" >> "$output_file"
        echo -e "${RED}  ❌ RUNTIME ERROR${NC}"
        FAILED_TESTS+=("$test_name")
        ((FAIL++)) || true
    else
        echo "FINAL STATUS: PASS" >> "$output_file"
        echo -e "${GREEN}  ✅ PASS${NC}"
        ((PASS++)) || true
    fi

    echo ""
    echo "Full output saved to: $output_file"
    echo ""
}

# Run all tests in repl directory
echo "Running tests in $TEST_DIR..."
echo ""
for test_file in "$TEST_DIR"/*.esk; do
    if [ -f "$test_file" ]; then
        run_test_verbose "$test_file"
    fi
done

# Create summary
echo "=========================================" | tee -a "$RESULTS_FILE"
echo "  REPL Test Results Summary" | tee -a "$RESULTS_FILE"
echo "=========================================" | tee -a "$RESULTS_FILE"
TOTAL=$((PASS + FAIL))
echo "Total Tests:    $TOTAL" | tee -a "$RESULTS_FILE"
echo -e "${GREEN}Passed:         $PASS${NC}" | tee -a "$RESULTS_FILE"
echo -e "${RED}Failed:         $FAIL${NC}" | tee -a "$RESULTS_FILE"
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

# Exit with appropriate code
if [ $FAIL -eq 0 ]; then
    exit 0
else
    exit 1
fi
