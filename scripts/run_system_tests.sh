#!/usr/bin/env bash

# System Test Suite (Hash Tables, File I/O, etc.)
# Runs all system-level tests


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
eshkol_test_isolation_init "system"
set +e  # Don't exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Honour $BUILD_DIR (CI passes it via the matrix); fall back to "build" for plain local runs.
BUILD_DIR="${BUILD_DIR:-build}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# Counters
PASS=0
FAIL=0

echo "========================================="
echo "  System Test Suite"
echo "========================================="
echo ""

TEST_DIR="tests/system"

if [ ! -d "$TEST_DIR" ]; then
    echo -e "${RED}Test directory not found: $TEST_DIR${NC}"
    exit 1
fi

for test_file in "$TEST_DIR"/*.esk; do
    if [ ! -f "$test_file" ]; then
        continue
    fi

    test_name=$(basename "$test_file")
    printf "Testing: %-40s " "$test_name"

    # Compile
    if ! ./$BUILD_DIR/eshkol-run "$test_file" -o "$ESHKOL_TEST_BIN" > /dev/null 2>&1; then
        echo -e "${RED}COMPILE FAIL${NC}"
        ((FAIL++))
        continue
    fi

    # Run.
    #
    # The output is captured, not discarded: these programs print their own
    # verdicts and exit 0 either way, so throwing stdout at /dev/null and
    # trusting the exit status certified every failing assertion as a PASS.
    if "$ESHKOL_TEST_BIN" > "$ESHKOL_TEST_OUT" 2>&1; then
        if eshkol_test_output_has_failure "$ESHKOL_TEST_OUT" 'error:'; then
            echo -e "${RED}ASSERTION FAIL${NC}"
            eshkol_test_output_failures "$ESHKOL_TEST_OUT" 'error:' 10 | sed 's/^/    /'
            ((FAIL++))
        else
            echo -e "${GREEN}PASS${NC}"
            ((PASS++))
        fi
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 139 ] || [ $EXIT_CODE -eq 134 ]; then
            echo -e "${RED}SEGFAULT${NC}"
        else
            echo -e "${RED}RUNTIME FAIL (exit $EXIT_CODE)${NC}"
        fi
        ((FAIL++))
    fi
done

# Summary
echo ""
echo "========================================="
TOTAL=$((PASS + FAIL))
echo "Total: $TOTAL  Passed: $PASS  Failed: $FAIL"
if [ $TOTAL -gt 0 ]; then
    PASS_RATE=$((PASS * 100 / TOTAL))
    echo "Pass Rate: ${PASS_RATE}%"
fi
echo "========================================="

eshkol_test_reset_bin
if [ $FAIL -gt 0 ]; then
    exit 1
else
    exit 0
fi
