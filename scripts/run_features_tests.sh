#!/bin/bash

# Eshkol Features Test Suite
# Runs all feature tests and reports results

set -e

# Per-run, per-repo-root isolation for temp files and build artifacts.
# Two suites (two worktrees, two agents, CI plus a local run) must never share
# a scratch path or a build artifact — see scripts/lib/test_isolation.sh.
# Sourcing must be checked *before* the fact: bash 3.2 (macOS) exits the
# shell when `source` cannot find its file, so a trailing `|| {...}` never
# runs there. A suite with no prelude has no failure detection and no
# scratch isolation, and must refuse to run rather than report a PASS.
ESHKOL_TEST_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/test_isolation.sh"
if [ ! -r "$ESHKOL_TEST_LIB" ]; then
    echo "FATAL: cannot read $ESHKOL_TEST_LIB" >&2
    echo "       (the shared test isolation and failure-detection prelude)." >&2
    echo "       Refusing to run: without it this suite would report a" >&2
    echo "       meaningless PASS." >&2
    exit 2
fi
source "$ESHKOL_TEST_LIB"
eshkol_test_isolation_init "features"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASS=0
FAIL=0
COMPILE_FAIL=0

# Results array
declare -a FAILED_TESTS
declare -a RUNTIME_ERRORS

echo "========================================="
echo "  Eshkol Features Test Suite"
echo "========================================="
echo ""

# Honour $BUILD_DIR (CI passes it via the matrix); fall back to "build" for plain local runs.
BUILD_DIR="${BUILD_DIR:-build}"
FAILURE_LINES="${ESHKOL_TEST_FAILURE_LINES:-40}"

# Ensure build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: build directory not found. Run cmake first.${NC}"
    exit 1
fi

# Check if compiler exists
if [ ! -f "$BUILD_DIR/eshkol-run" ]; then
    echo -e "${RED}Error: eshkol-run not found. Run make first.${NC}"
    exit 1
fi

# Pin the compiler for the duration of the run.
#
# This is the longest of the per-directory suites and nothing used to hold the
# binary still while it ran: a `cmake --build` in the same worktree swapped
# eshkol-run, stdlib.o and the runtime archive underneath it, and the resulting
# "failures" belonged to no single build. Copy the toolchain into this run's own
# scratch directory and drive that copy, so a concurrent relink is irrelevant.
# The copy carries stdlib.o/stdlib.bc and libeshkol-runtime.a, which is what
# makes the pin hold for the AOT link too (see the helper's notes on
# find_runtime_library's exe-dir precedence) — hence -L at the pinned dir rather
# than the live build tree.
PINNED_BUILD_DIR="$(eshkol_test_pin_toolchain "$BUILD_DIR")"
ESHKOL_RUN_BIN="$PINNED_BUILD_DIR/eshkol-run"
echo "Pinned toolchain for this run: $PINNED_BUILD_DIR"

echo "Testing all files in tests/features/ directory..."
echo ""

# Run each test
for test_file in tests/features/*.esk; do
    test_name=$(basename "$test_file")
    printf "Testing %-50s " "$test_name"

    # Clean up stale temp files before each test
    rm -f "$ESHKOL_TEST_BIN" "$ESHKOL_TEST_BIN.tmp.o" "$ESHKOL_TEST_OUT" "$ESHKOL_TEST_COMPILE_LOG"

    # Try to compile
    if "$ESHKOL_RUN_BIN" -L"$PINNED_BUILD_DIR" "$test_file" -o "$ESHKOL_TEST_BIN" > "$ESHKOL_TEST_COMPILE_LOG" 2>&1; then
        # Compilation succeeded, try to run
        if "$ESHKOL_TEST_BIN" > "$ESHKOL_TEST_OUT" 2>&1; then
            # A zero exit status is not a pass. These tests print their own
            # verdicts and exit 0 regardless, so scan the output for failure
            # markers as well — anywhere on the line, not just at column 0.
            if eshkol_test_output_has_failure "$ESHKOL_TEST_OUT" 'error:'; then
                echo -e "${YELLOW}⚠ RUNTIME ERROR${NC}"
                eshkol_test_output_failures "$ESHKOL_TEST_OUT" 'error:' "$FAILURE_LINES" | sed 's/^/    /'
                RUNTIME_ERRORS+=("$test_name")
                ((FAIL++)) || true
            elif eshkol_test_output_is_silent "$ESHKOL_TEST_OUT"; then
                # Printed nothing at all: absence of a verdict is not a pass.
                echo -e "${RED}❌ NO OUTPUT${NC}"
                FAILED_TESTS+=("$test_name")
                ((FAIL++)) || true
            else
                echo -e "${GREEN}✅ PASS${NC}"
                ((PASS++)) || true
            fi
        else
            echo -e "${RED}❌ RUNTIME FAIL${NC}"
            head -n "$FAILURE_LINES" "$ESHKOL_TEST_OUT" | sed 's/^/    /'
            FAILED_TESTS+=("$test_name")
            ((FAIL++)) || true
        fi
    else
        echo -e "${RED}❌ COMPILE FAIL${NC}"
        head -n "$FAILURE_LINES" "$ESHKOL_TEST_COMPILE_LOG" | sed 's/^/    /'
        FAILED_TESTS+=("$test_name")
        ((COMPILE_FAIL++)) || true
        ((FAIL++)) || true
    fi
done

echo ""
echo "========================================="
echo "  Test Results Summary"
echo "========================================="
echo -e "Total Tests:    $(( PASS + FAIL ))"
echo -e "${GREEN}Passed:         $PASS${NC}"
echo -e "${RED}Failed:         $FAIL${NC}"
echo -e "  Compile Failures: $COMPILE_FAIL"
echo -e "  Runtime Errors:   ${#RUNTIME_ERRORS[@]}"
echo ""

if [ $FAIL -gt 0 ]; then
    echo "Failed Tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo ""

    if [ ${#RUNTIME_ERRORS[@]} -gt 0 ]; then
        echo "Runtime Errors:"
        for test in "${RUNTIME_ERRORS[@]}"; do
            echo "  - $test"
        done
        echo ""
    fi
fi

# Calculate pass rate
TOTAL=$(( PASS + FAIL ))
if [ $TOTAL -gt 0 ]; then
    PASS_RATE=$(( PASS * 100 / TOTAL ))
    echo "Pass Rate: ${PASS_RATE}%"
fi

echo ""

# Clean up
rm -f "$ESHKOL_TEST_OUT" "$ESHKOL_TEST_COMPILE_LOG" "$ESHKOL_TEST_BIN"

# Exit with appropriate code
if [ $FAIL -eq 0 ]; then
    exit 0
else
    exit 1
fi
