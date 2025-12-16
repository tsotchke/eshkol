#!/bin/bash

# Eshkol Complete Test Suite (Verbose Output)
# Runs all test suites with full output displayed

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Counters
SUITES_PASS=0
SUITES_FAIL=0

# Results arrays
declare -a PASSED_SUITES
declare -a FAILED_SUITES

# Test scripts to run (in order) - using output versions for verification
TEST_SCRIPTS=(
    "run_features_tests_with_output.sh"
    "run_stdlib_tests_with_output.sh"
    "run_list_tests_with_output.sh"
    "run_memory_tests_with_output.sh"
    "run_modules_tests_with_output.sh"
    "run_types_tests_with_output.sh"
    "run_autodiff_tests_with_output.sh"
    "run_ml_tests_with_output.sh"
    "run_neural_tests_with_output.sh"
    "run_json_tests_with_output.sh"
    "run_system_tests_with_output.sh"
    "run_cpp_type_tests.sh"
)

echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}   Eshkol Complete Test Suite${NC}"
echo -e "${BLUE}   (Verbose Output Mode)${NC}"
echo -e "${BLUE}=========================================${NC}"
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

echo "Running all test suites with full output..."
echo ""

# Create temp file for output
OUTPUT_LOG=$(mktemp)
trap "rm -f $OUTPUT_LOG" EXIT

# Run each test suite
for script in "${TEST_SCRIPTS[@]}"; do
    script_path="$SCRIPT_DIR/$script"
    suite_name="${script%.sh}"
    suite_name="${suite_name#run_}"
    suite_name="${suite_name%_tests}"

    if [ ! -f "$script_path" ]; then
        echo -e "${YELLOW}⚠ Skipping $script (not found)${NC}"
        continue
    fi

    echo ""
    echo -e "${BLUE}═════════════════════════════════════════${NC}"
    echo -e "${BLUE}   Running: $suite_name tests${NC}"
    echo -e "${BLUE}═════════════════════════════════════════${NC}"
    echo ""

    # Run with full output, capture exit code
    if bash "$script_path" 2>&1 | tee "$OUTPUT_LOG"; then
        echo ""
        echo -e "${GREEN}═══ $suite_name: PASSED ═══${NC}"
        PASSED_SUITES+=("$suite_name")
        ((SUITES_PASS++))
    else
        echo ""
        echo -e "${RED}═══ $suite_name: FAILED ═══${NC}"
        FAILED_SUITES+=("$suite_name")
        ((SUITES_FAIL++))
    fi
done

echo ""
echo ""
echo -e "${BLUE}═════════════════════════════════════════${NC}"
echo -e "${BLUE}   Complete Test Suite Summary${NC}"
echo -e "${BLUE}═════════════════════════════════════════${NC}"
echo ""
echo -e "Total Suites:   $(( SUITES_PASS + SUITES_FAIL ))"
echo -e "${GREEN}Passed:         $SUITES_PASS${NC}"
echo -e "${RED}Failed:         $SUITES_FAIL${NC}"
echo ""

if [ ${#PASSED_SUITES[@]} -gt 0 ]; then
    echo -e "${GREEN}Passed Suites:${NC}"
    for suite in "${PASSED_SUITES[@]}"; do
        echo -e "  ${GREEN}✅ $suite${NC}"
    done
    echo ""
fi

if [ ${#FAILED_SUITES[@]} -gt 0 ]; then
    echo -e "${RED}Failed Suites:${NC}"
    for suite in "${FAILED_SUITES[@]}"; do
        echo -e "  ${RED}❌ $suite${NC}"
    done
    echo ""
fi

# Calculate pass rate
TOTAL=$(( SUITES_PASS + SUITES_FAIL ))
if [ $TOTAL -gt 0 ]; then
    PASS_RATE=$(( SUITES_PASS * 100 / TOTAL ))
    echo "Suite Pass Rate: ${PASS_RATE}%"
fi

echo ""

# Exit with appropriate code
if [ $SUITES_FAIL -eq 0 ]; then
    echo -e "${GREEN}All test suites passed!${NC}"
    exit 0
else
    echo -e "${RED}Some test suites failed.${NC}"
    exit 1
fi
