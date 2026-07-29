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

# Per-run, per-repo-root isolation. This aggregator owns its own EXIT trap, so
# it opts out of the helper's and calls the cleanup from its own.
ESHKOL_TEST_ISOLATION_NO_TRAP=1
# Sourcing must be checked *before* the fact: bash 3.2 (macOS) exits the
# shell when `source` cannot find its file, so a trailing `|| {...}` never
# runs there. A suite with no prelude has no failure detection and no
# scratch isolation, and must refuse to run rather than report a PASS.
ESHKOL_TEST_LIB="$SCRIPT_DIR/lib/test_isolation.sh"
if [ ! -r "$ESHKOL_TEST_LIB" ]; then
    echo "FATAL: cannot read $ESHKOL_TEST_LIB" >&2
    echo "       (the shared test isolation and failure-detection prelude)." >&2
    echo "       Refusing to run: without it this suite would report a" >&2
    echo "       meaningless PASS." >&2
    exit 2
fi
source "$ESHKOL_TEST_LIB"
eshkol_test_isolation_init "all"

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

cleanup_tests_tmpdir() {
    # TMPDIR_TESTS lives inside this run's private scratch directory, which the
    # shared helper owns and removes; nothing here reaches outside it.
    eshkol_test_isolation_cleanup
}

require_regular_executable() {
    local label="$1"
    local path="$2"

    if [ -L "$path" ] || ! test -f "$path" || ! test -s "$path" || ! test -x "$path"; then
        echo -e "${RED}Error: $label missing, empty, symlinked, or not executable: $path${NC}"
        exit 1
    fi
}

validate_suite_script() {
    local path="$1"

    if ! test -e "$path"; then
        return 1
    fi

    if [ -L "$path" ] || ! test -f "$path" || ! test -s "$path"; then
        echo -e "${RED}Error: suite script missing, empty, symlinked, or not a regular file: $path${NC}"
        return 2
    fi

    return 0
}

run_suite_script() {
    local path="$1"
    local status

    validate_suite_script "$path"
    status=$?
    if [ "$status" -ne 0 ]; then
        return 126
    fi

    bash "$path"
}

# Temp directory for captured output — inside this run's scratch dir, so two
# aggregators (two worktrees, or CI beside a local run) cannot read each other's
# suite logs.
TMPDIR_TESTS="$ESHKOL_TEST_TMPDIR/suite-logs"
mkdir -p "$TMPDIR_TESTS"
trap cleanup_tests_tmpdir EXIT

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
    "run_manifold_tests.sh"
    "run_ml_tests.sh"
    "run_neural_tests.sh"
    "run_json_tests.sh"
    "run_system_tests.sh"
    "run_complex_tests.sh"
    "run_cpp_type_tests.sh"
    "run_vm_tests.sh"
    "run_vm_surface_tests.sh"
    "run_r7rs_tests.sh"
    "run_surface_extension_tests.sh"
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
    "run_ffi_tests.sh"
    "run_benchmark_tests.sh"
    "run_migration_tests.sh"
    "run_codegen_tests.sh"
    "run_codegen_optlevel_tests.sh"
    "run_numeric_tests.sh"
    "test_run_sicp_smoke_gate.sh"
    "run_v1_2_edge_cases_tests.sh"
)

echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}   Eshkol Complete Test Suite${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# Honour $BUILD_DIR (CI passes it via the matrix); fall back to "build" for plain local runs.
BUILD_DIR="${BUILD_DIR:-build}"

# Ensure build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: build directory not found. Run cmake first.${NC}"
    exit 1
fi

# Check if compiler exists
require_regular_executable "eshkol-run" "$BUILD_DIR/eshkol-run"

# Pin the compiler for the duration of the run.
#
# This aggregator runs for tens of minutes and delegates to sub-suites whose
# BUILD_DIR contract is repository-relative (`./$BUILD_DIR/eshkol-run`), so we
# cannot redirect them at a private copy without changing that interface. What
# we can do is refuse to report results gathered across a rebuild: a run that
# straddled two relinks once reported "93% pass, 6 failures including SEGFAULT
# in examples/autodiff.esk" — every one of which passes on a stable build. The
# crash was the harness inventing a failure, and a harness that can do that
# cannot certify anything. Fingerprint the relinkable artifacts now, re-check
# at the end, and if they moved, say so instead of printing a verdict.
eshkol_test_toolchain_snapshot "$BUILD_DIR"

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
    # The colon is NOT required and the marker is NOT anchored: test programs
    # print `  <case>: FAIL` (indented, bare FAIL, no colon after it) as often
    # as they print `FAIL: <case>`. Requiring `^\s*FAIL:` here hid whole classes
    # of assertion failure — tests/gpu/sf64_primitives_test.esk being the case
    # that exposed it.
    # Filter the whole log once — zero-count summaries and decorative titles
    # out, "Testing foo.esk" result lines out (Pattern 1 owns those) — then take
    # what is left. Filtering per line would fork two processes per log line.
    local assert_file="$TMPDIR_TESTS/${suite_name}_assertions.log"
    grep -v '\.esk' "$clean_file" 2>/dev/null \
        | eshkol_test_filter_verdict_noise \
        | grep -E '(^|[^A-Za-z0-9_])FAIL([^A-Za-z0-9_]|$)' \
        > "$assert_file" 2>/dev/null || true
    while IFS= read -r line; do
        desc=$(printf '%s' "$line" | sed -E 's/^[[:space:]]*//; s/[[:space:]]+$//')
        [ -n "$desc" ] || continue
        ALL_FAILURES+=("$suite_name: $desc (ASSERTION)")
    done < "$assert_file"

    # Count passes and fails from suite summary lines.
    # Handles:
    #   "Passed: N"
    #   "Working: N"
    #   "Failed: N"
    #   "Results: N passed, M failed"
    local suite_passed=$(grep -oE '(Passed|Working):[[:space:]]+[0-9]+' "$clean_file" | tail -1 | grep -oE '[0-9]+' || echo 0)
    local suite_failed=$(grep -oE 'Failed:[[:space:]]+[0-9]+' "$clean_file" | tail -1 | grep -oE '[0-9]+' || echo 0)
    if [ -z "$suite_passed" ]; then suite_passed=0; fi
    if [ -z "$suite_failed" ]; then suite_failed=0; fi

    if [ "$suite_passed" -eq 0 ]; then
        local results_passed=$(grep -oE 'Results:[[:space:]]+[0-9]+[[:space:]]+passed' "$clean_file" | tail -1 | grep -oE '[0-9]+' || echo 0)
        if [ -n "$results_passed" ]; then
            suite_passed=$results_passed
        fi
    fi

    if [ "$suite_failed" -eq 0 ]; then
        local results_failed=$(grep -oE 'Results:[[:space:]]+[0-9]+[[:space:]]+passed,[[:space:]]+[0-9]+[[:space:]]+failed' "$clean_file" | tail -1 | sed -E 's/.*passed,[[:space:]]+([0-9]+)[[:space:]]+failed.*/\1/' || echo 0)
        if [ -n "$results_failed" ]; then
            suite_failed=$results_failed
        fi
    fi

    TOTAL_TESTS_PASS=$(( TOTAL_TESTS_PASS + suite_passed ))
    TOTAL_TESTS_FAIL=$(( TOTAL_TESTS_FAIL + suite_failed ))
}

# Run each test suite
for script in "${TEST_SCRIPTS[@]}"; do
    script_path="$SCRIPT_DIR/$script"
    suite_name="${script%.sh}"
    suite_name="${suite_name#run_}"
    suite_name="${suite_name%_tests}"

    validate_suite_script "$script_path"
    suite_script_status=$?
    if [ "$suite_script_status" -eq 1 ]; then
        echo -e "${YELLOW}-- Skipping $script (not found)${NC}"
        SKIPPED_SUITES+=("$suite_name")
        ((SUITES_SKIP++)) || true
        continue
    fi
    if [ "$suite_script_status" -ne 0 ]; then
        FAILED_SUITES+=("$suite_name")
        ((SUITES_FAIL++)) || true
        continue
    fi

    echo -e "${BLUE}─────────────────────────────────────────${NC}"
    echo -e "${BLUE}Running: $suite_name tests${NC}"
    echo -e "${BLUE}─────────────────────────────────────────${NC}"

    # Capture output while still displaying it
    output_file="$TMPDIR_TESTS/${suite_name}.log"
    run_suite_script "$script_path" 2>&1 | tee "$output_file"
    suite_exit=${PIPESTATUS[0]}

    # Absence of output is not a pass. Every suite prints a banner, so an empty
    # log means the script died before it produced any evidence — and a suite
    # that reported nothing must not be scored as one that reported success.
    if [ $suite_exit -eq 0 ] && eshkol_test_output_is_silent "$output_file"; then
        echo -e "${RED}>>> $suite_name: FAILED (exited 0 but produced no output)${NC}"
        FAILED_SUITES+=("$suite_name")
        ALL_FAILURES+=("$suite_name: <no output> (SILENT SUITE)")
        ((SUITES_FAIL++)) || true
        echo ""
        continue
    fi

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

# The compiler must not have changed underneath the run. If it did, no verdict
# from this run is trustworthy — including the pass count printed above — so
# exit on a distinct code rather than letting either 0 or 1 be believed.
if ! eshkol_test_toolchain_verify "$BUILD_DIR"; then
    exit 3
fi

# Exit with appropriate code.
#
# SUITES_FAIL comes from child exit codes. UNIQUE_FAILURES comes from scraping
# the suite output. When those two disagree — individual failures were printed
# yet every suite exited 0 — the aggregator used to print "ALL FAILING TESTS
# (N total)" and then exit 0, which is the exact false-PASS this pass exists to
# remove. Treat the contradiction as a failure of the suite that produced it.
if [ $SUITES_FAIL -eq 0 ] && [ ${#UNIQUE_FAILURES[@]} -gt 0 ]; then
    echo -e "${RED}${BOLD}Harness contradiction: ${#UNIQUE_FAILURES[@]} individual test failure(s) were" \
            "reported above, but every suite exited 0.${NC}"
    echo -e "${RED}A suite is printing failures and returning success. Failing the run:" \
            "an unreported failure is worse than a red build.${NC}"
    echo ""
    exit 1
fi

if [ $SUITES_FAIL -eq 0 ]; then
    exit 0
else
    exit 1
fi
