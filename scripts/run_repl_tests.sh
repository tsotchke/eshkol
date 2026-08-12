#!/usr/bin/env bash

# Eshkol REPL Test Suite
# Runs all REPL tests and reports results

set -e

# Honour $BUILD_DIR (CI passes it via the matrix); fall back to "build" for plain local runs.
BUILD_DIR="${BUILD_DIR:-build}"
REPL_TEST_TIMEOUT="${REPL_TEST_TIMEOUT:-30}"

if ! [[ "$REPL_TEST_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: REPL_TEST_TIMEOUT must be a positive integer number of seconds." >&2
    exit 2
fi

if [[ "$BUILD_DIR" = /* ]]; then
    REPL_BIN="$BUILD_DIR/eshkol-repl"
else
    REPL_BIN="./$BUILD_DIR/eshkol-repl"
fi

# Per-run, per-repo-root isolation for temp files and build artifacts.
# Two suites (two worktrees, two agents, CI plus a local run) must never share
# a scratch path — see scripts/lib/test_isolation.sh.
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
eshkol_test_isolation_init "repl"
OUTPUT_FILE="$ESHKOL_TEST_TMPDIR/repl.out"


# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASS=0
FAIL=0

# Results array
declare -a FAILED_TESTS

echo "========================================="
echo "  Eshkol REPL Test Suite"
echo "========================================="
echo ""

# Ensure build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: build directory '$BUILD_DIR' not found. Run cmake first.${NC}"
    exit 1
fi

# Check if REPL exists
if [ ! -x "$REPL_BIN" ]; then
    echo -e "${RED}Error: executable eshkol-repl not found under '$BUILD_DIR'. Run the build first.${NC}"
    exit 1
fi

# Check if test directory exists
if [ ! -d "tests/repl" ]; then
    echo -e "${YELLOW}Warning: tests/repl directory not found. Creating...${NC}"
    mkdir -p tests/repl
fi

echo "Testing all files in tests/repl/ directory..."
echo ""

# Run each test
for test_file in tests/repl/*.esk; do
    # Skip if no files found
    [ -e "$test_file" ] || continue

    test_name=$(basename "$test_file")
    printf "Testing %-50s " "$test_name"

    # Run the test through REPL (add :quit at the end)
    # Use timeout command if available, otherwise just run directly
    set +e
    if command -v timeout > /dev/null 2>&1; then
        # Linux has timeout
        { cat "$test_file"; echo ""; echo ":quit"; } | timeout "$REPL_TEST_TIMEOUT" "$REPL_BIN" > "${OUTPUT_FILE:?}" 2>&1
        EXIT_CODE=$?
    else
        # macOS - run directly (no timeout needed, tests are fast)
        { cat "$test_file"; echo ""; echo ":quit"; } | "$REPL_BIN" > "${OUTPUT_FILE:?}" 2>&1
        EXIT_CODE=$?
    fi
    set -e

    # Check for errors in output
    if [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 142 ]; then
        echo -e "${YELLOW}⚠ TIMEOUT${NC}"
        FAILED_TESTS+=("$test_name (timeout)")
        ((FAIL++)) || true
    elif [ $EXIT_CODE -ne 0 ]; then
        echo -e "${RED}❌ PROCESS FAILURE (exit $EXIT_CODE)${NC}"
        FAILED_TESTS+=("$test_name (exit $EXIT_CODE)")
        ((FAIL++)) || true
    # `error:` alone is a compiler diagnostic, not a verdict: these
    # programs print their own FAIL lines and exit 0, so scan for
    # failure markers too — anywhere on the line, not just column 0.
    elif eshkol_test_output_has_failure "$OUTPUT_FILE" 'error:'; then
        echo -e "${RED}❌ RUNTIME ERROR${NC}"
        FAILED_TESTS+=("$test_name")
        ((FAIL++)) || true
    elif grep -q "Segmentation fault" "$OUTPUT_FILE" 2>/dev/null; then
        echo -e "${RED}❌ SEGFAULT${NC}"
        FAILED_TESTS+=("$test_name")
        ((FAIL++)) || true
    else
        echo -e "${GREEN}✅ PASS${NC}"
        ((PASS++)) || true
    fi
done

echo ""
echo "========================================="
echo "  Test Results Summary"
echo "========================================="
echo -e "Total Tests:    $(( PASS + FAIL ))"
echo -e "${GREEN}Passed:         $PASS${NC}"
echo -e "${RED}Failed:         $FAIL${NC}"
echo ""

if [ $FAIL -gt 0 ]; then
    echo "Failed Tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo ""
fi

# Calculate pass rate
TOTAL=$(( PASS + FAIL ))
if [ $TOTAL -gt 0 ]; then
    PASS_RATE=$(( PASS * 100 / TOTAL ))
    echo "Pass Rate: ${PASS_RATE}%"
fi

echo ""

# Exit with appropriate code
if [ $FAIL -eq 0 ]; then
    exit 0
else
    exit 1
fi
