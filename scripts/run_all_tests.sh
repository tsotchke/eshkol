#!/bin/bash

# Eshkol Complete Test Suite
# Runs all test suites and reports aggregate results
# Shows all individual failing tests at the bottom

# DO NOT use set -e — we need to continue after suite failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Counters
SUITES_PASS=0
SUITES_FAIL=0
SUITES_SKIP=0
TOTAL_TESTS_PASS=0
TOTAL_TESTS_FAIL=0

# Results arrays
declare -a PASSED_SUITES
declare -a FAILED_SUITES
declare -a SKIPPED_SUITES
declare -a ALL_FAILURES  # "suite: test_name (reason)" entries

# Temp directory for captured output
TMPDIR_TESTS=$(mktemp -d)
trap "rm -rf '$TMPDIR_TESTS'" EXIT

# Test scripts to run (in order)
TEST_SCRIPTS=(
    "run_features_tests.sh"
    "run_stdlib_tests.sh"
    "run_list_tests.sh"
    "run_memory_tests.sh"
    "run_modules_tests.sh"
    "run_types_tests.sh"
    "run_typesystem_tests.sh"
    "run_autodiff_tests.sh"
    "run_ml_tests.sh"
    "run_neural_tests.sh"
    "run_json_tests.sh"
    "run_system_tests.sh"
    "run_complex_tests.sh"
    "run_cpp_type_tests.sh"
    "run_parser_tests.sh"
    "run_control_flow_tests.sh"
    "run_logic_tests.sh"
    "run_bignum_tests.sh"
    "run_rational_tests.sh"
    "run_parallel_tests.sh"
    "run_signal_tests.sh"
    "run_optimization_tests.sh"
    "run_examples_tests.sh"
    "run_xla_tests.sh"
    "run_gpu_tests.sh"
    "run_error_handling_tests.sh"
    "run_macros_tests.sh"
    "run_repl_tests.sh"
    "run_web_tests.sh"
    "run_tco_tests.sh"
    "run_io_tests.sh"
    "run_benchmark_tests.sh"
    "run_migration_tests.sh"
    "run_codegen_tests.sh"
    "run_numeric_tests.sh"
)

echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}   Eshkol Complete Test Suite${NC}"
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

echo "Running all test suites..."
echo ""

# Extract individual test failures from captured suite output
extract_failures() {
    local suite_name="$1"
    local output_file="$2"

    # Strip ANSI escape codes once into a clean temp file
    local clean_file="$TMPDIR_TESTS/${suite_name}_clean.log"
    sed 's/\x1b\[[0-9;]*m//g' "$output_file" > "$clean_file"

    # Pattern 1: Test result lines — a .esk filename on the same line as a failure keyword
    # Matches all known formats across every test script:
    #   "Testing some_test.esk                  COMPILE FAIL"
    #   "Testing some_test.esk                  RUNTIME FAIL"
    #   "Testing some_test.esk                  RUNTIME FAIL (exit 139)"
    #   "Testing some_test.esk                  RUNTIME ERROR"
    #   "Testing some_test.esk                  ASSERTION FAIL"
    #   "Testing some_test.esk                  TESTS FAILED"
    #   "Testing some_test.esk                  SEGFAULT"
    #   "Testing some_test.esk                  FAIL"
    #   "[  1/  5] some_test.esk                RUNTIME FAIL (exit 1)"
    #
    # EDGE CASE: Segfaults can split the output — the shell prints the crash
    # message between printf and the echo, so the .esk filename and the
    # failure keyword end up on SEPARATE lines:
    #   "Testing some_test.esk                  <segfault message>"
    #   "RUNTIME FAIL (exit 139)"
    # For these, we track the last-seen .esk filename and use it.
    #
    # NOTE: No \b word boundaries — "TESTS FAILED" must match even though
    # "FAILED" has no boundary after "FAIL". Order matters for grep -oE:
    # longer patterns first so "COMPILE FAIL" matches before bare "FAIL".
    local FAIL_PATTERN='(COMPILE FAIL|RUNTIME FAIL|RUNTIME ERROR|ASSERTION FAIL|TESTS FAILED|SEGFAULT|FAIL)'
    local last_test_file=""
    while IFS= read -r line; do
        # Track the most recent .esk filename we've seen
        local line_esk=$(echo "$line" | grep -oE '[A-Za-z0-9_/.-]+\.esk' | head -1)
        if [ -n "$line_esk" ]; then
            last_test_file="$line_esk"
        fi

        # Check if this line has a failure keyword
        # Skip summary lines like "Failed: 0", "Failed Tests:", "Compile Failures: 2"
        local fail_type=""
        if ! echo "$line" | grep -qE '^\s*(Failed|Passed|Total|Compile Failures|Runtime|Pass Rate|Some .* failed|Fix these)'; then
            fail_type=$(echo "$line" | grep -oE "$FAIL_PATTERN" | head -1)
        fi
        if [ -n "$fail_type" ]; then
            # Use .esk from this line if present, otherwise use last-seen
            local matched_file="${line_esk:-$last_test_file}"
            if [ -n "$matched_file" ]; then
                ALL_FAILURES+=("$suite_name: $matched_file ($fail_type)")
            fi
        fi
    done < "$clean_file"

    # Pattern 2: "FAIL: description" assertion lines printed by test programs
    # These appear on their own lines, NOT on the "Testing foo.esk" line
    # e.g. "FAIL: Accumulator pattern: build list of 1000 elements"
    while IFS= read -r line; do
        if echo "$line" | grep -qE '^\s*FAIL:'; then
            desc=$(echo "$line" | sed -E 's/^\s*FAIL:[[:space:]]*//')
            ALL_FAILURES+=("$suite_name: $desc (ASSERTION)")
        fi
    done < "$clean_file"

    # Count passes and fails from suite summary lines
    # Handles: "Passed: N", "Working: N", "Failed: N", "Compile Failures: N"
    local suite_passed=$(grep -oE '(Passed|Working):[[:space:]]+[0-9]+' "$clean_file" | tail -1 | grep -oE '[0-9]+' || echo 0)
    local suite_failed=$(grep -oE 'Failed:[[:space:]]+[0-9]+' "$clean_file" | tail -1 | grep -oE '[0-9]+' || echo 0)
    if [ -z "$suite_passed" ]; then suite_passed=0; fi
    if [ -z "$suite_failed" ]; then suite_failed=0; fi
    TOTAL_TESTS_PASS=$(( TOTAL_TESTS_PASS + suite_passed ))
    TOTAL_TESTS_FAIL=$(( TOTAL_TESTS_FAIL + suite_failed ))
}

# Run each test suite
for script in "${TEST_SCRIPTS[@]}"; do
    script_path="$SCRIPT_DIR/$script"
    suite_name="${script%.sh}"
    suite_name="${suite_name#run_}"
    suite_name="${suite_name%_tests}"

    if [ ! -f "$script_path" ]; then
        echo -e "${YELLOW}-- Skipping $script (not found)${NC}"
        SKIPPED_SUITES+=("$suite_name")
        ((SUITES_SKIP++)) || true
        continue
    fi

    echo -e "${BLUE}─────────────────────────────────────────${NC}"
    echo -e "${BLUE}Running: $suite_name tests${NC}"
    echo -e "${BLUE}─────────────────────────────────────────${NC}"

    # Capture output while still displaying it
    output_file="$TMPDIR_TESTS/${suite_name}.log"
    bash "$script_path" 2>&1 | tee "$output_file"
    suite_exit=${PIPESTATUS[0]}

    if [ $suite_exit -eq 0 ]; then
        echo -e "${GREEN}>>> $suite_name: PASSED${NC}"
        PASSED_SUITES+=("$suite_name")
        ((SUITES_PASS++)) || true
    else
        echo -e "${RED}>>> $suite_name: FAILED${NC}"
        FAILED_SUITES+=("$suite_name")
        ((SUITES_FAIL++)) || true
    fi

    # Always extract — captures individual test results and counts for ALL suites
    extract_failures "$suite_name" "$output_file"
    echo ""
done

# Deduplicate ALL_FAILURES (Pattern 1 and Pattern 2 can overlap)
# Use a newline-delimited seen list (bash 3.2 compatible — no associative arrays)
declare -a UNIQUE_FAILURES
_SEEN_LIST=""
for f in "${ALL_FAILURES[@]}"; do
    # Sanitize to a comparable key
    key="${f//[^a-zA-Z0-9_]/_}"
    case "$_SEEN_LIST" in
        *"|$key|"*) ;;  # already seen
        *)
            _SEEN_LIST="${_SEEN_LIST}|${key}|"
            UNIQUE_FAILURES+=("$f")
            ;;
    esac
done

echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}   Complete Test Suite Summary${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
TOTAL_SUITES=$(( SUITES_PASS + SUITES_FAIL ))
TOTAL_INDIVIDUAL=$(( TOTAL_TESTS_PASS + TOTAL_TESTS_FAIL ))
echo -e "Total Suites Run:   $TOTAL_SUITES"
echo -e "Suites Skipped:     $SUITES_SKIP"
echo -e "${GREEN}Suites Passed:      $SUITES_PASS${NC}"
echo -e "${RED}Suites Failed:      $SUITES_FAIL${NC}"
echo ""
if [ $TOTAL_INDIVIDUAL -gt 0 ]; then
    INDIVIDUAL_RATE=$(( TOTAL_TESTS_PASS * 100 / TOTAL_INDIVIDUAL ))
    echo -e "Individual Tests:   $TOTAL_INDIVIDUAL"
    echo -e "${GREEN}  Passed:           $TOTAL_TESTS_PASS${NC}"
    echo -e "${RED}  Failed:           $TOTAL_TESTS_FAIL${NC}"
    echo -e "  Pass Rate:        ${INDIVIDUAL_RATE}%"
    echo ""
fi

if [ ${#PASSED_SUITES[@]} -gt 0 ]; then
    echo -e "${GREEN}Passed Suites:${NC}"
    for suite in "${PASSED_SUITES[@]}"; do
        echo -e "  ${GREEN}+ $suite${NC}"
    done
    echo ""
fi

if [ ${#SKIPPED_SUITES[@]} -gt 0 ]; then
    echo -e "${YELLOW}Skipped Suites (script not found):${NC}"
    for suite in "${SKIPPED_SUITES[@]}"; do
        echo -e "  ${YELLOW}~ $suite${NC}"
    done
    echo ""
fi

if [ ${#FAILED_SUITES[@]} -gt 0 ]; then
    echo -e "${RED}Failed Suites:${NC}"
    for suite in "${FAILED_SUITES[@]}"; do
        echo -e "  ${RED}X $suite${NC}"
    done
    echo ""
fi

# Calculate pass rate
TOTAL=$(( SUITES_PASS + SUITES_FAIL ))
if [ $TOTAL -gt 0 ]; then
    PASS_RATE=$(( SUITES_PASS * 100 / TOTAL ))
    echo "Suite Pass Rate: ${PASS_RATE}%"
    echo ""
fi

# ===== THE KEY PART: Individual failing tests at the bottom =====
if [ ${#UNIQUE_FAILURES[@]} -gt 0 ]; then
    echo -e "${RED}${BOLD}=========================================${NC}"
    echo -e "${RED}${BOLD}   ALL FAILING TESTS (${#UNIQUE_FAILURES[@]} total)${NC}"
    echo -e "${RED}${BOLD}=========================================${NC}"
    echo ""
    for failure in "${UNIQUE_FAILURES[@]}"; do
        echo -e "  ${RED}X $failure${NC}"
    done
    echo ""
    echo -e "${RED}Fix these ${#UNIQUE_FAILURES[@]} test(s) to reach 100% pass rate.${NC}"
else
    if [ $SUITES_FAIL -eq 0 ]; then
        echo -e "${GREEN}${BOLD}All test suites passed!${NC}"
    else
        echo -e "${RED}Some suites failed but no individual test failures were extracted.${NC}"
        echo -e "${RED}Check the suite output above for details.${NC}"
    fi
fi

echo ""

# Exit with appropriate code
if [ $SUITES_FAIL -eq 0 ]; then
    exit 0
else
    exit 1
fi
