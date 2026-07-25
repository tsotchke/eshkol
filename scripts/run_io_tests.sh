#!/bin/bash

# Eshkol I/O Test Suite
# Runs all I/O tests (read, write, string ports)

set -e

# Per-run, per-repo-root isolation for temp files and build artifacts.
# Two suites (two worktrees, two agents, CI plus a local run) must never share
# a scratch path or a build artifact — see scripts/lib/test_isolation.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/test_isolation.sh"
eshkol_test_isolation_init "io"

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

run_with_timeout() {
    local seconds="$1"
    shift

    "$@" &
    local cmd_pid=$!

    (
        sleep "$seconds"
        if kill -0 "$cmd_pid" 2>/dev/null; then
            kill "$cmd_pid" 2>/dev/null || true
        fi
    ) &
    local watchdog_pid=$!

    wait "$cmd_pid"
    local rc=$?

    kill "$watchdog_pid" 2>/dev/null || true
    wait "$watchdog_pid" 2>/dev/null || true

    return "$rc"
}

echo "========================================="
echo "  Eshkol I/O Test Suite"
echo "========================================="
echo ""

# Honour $BUILD_DIR (CI passes it via the matrix); fall back to "build" for plain local runs.
BUILD_DIR="${BUILD_DIR:-build}"

# Ensure build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: build directory not found. Run cmake first.${NC}"
    exit 1
fi

if [ ! -f "$BUILD_DIR/eshkol-run" ]; then
    echo -e "${RED}Error: eshkol-run not found. Run make first.${NC}"
    exit 1
fi

echo "Testing all files in tests/io/ directory..."
echo ""

for test_file in tests/io/*.esk; do
    [ -f "$test_file" ] || continue
    test_name=$(basename "$test_file")
    printf "Testing %-50s " "$test_name"

    eshkol_test_reset_bin
    if ./$BUILD_DIR/eshkol-run -L./$BUILD_DIR "$test_file" -o "$ESHKOL_TEST_BIN" > /dev/null 2>&1; then
        # Use timeout to prevent hangs (some IO tests may block on unimplemented port reads)
        if run_with_timeout 10 "$ESHKOL_TEST_BIN" > "$ESHKOL_TEST_OUT" 2>&1; then
            # A failure marker anywhere in the output fails the test — the old
            # `^FAIL`-anchored match never saw the indented `  <case>: FAIL`
            # form that most test programs actually print.
            if eshkol_test_output_has_failure "$ESHKOL_TEST_OUT"; then
                echo -e "${RED}❌ FAIL${NC}"
                FAILED_TESTS+=("$test_name")
                ((FAIL++)) || true
            else
                echo -e "${GREEN}✅ PASS${NC}"
                ((PASS++)) || true
            fi
        else
            echo -e "${RED}❌ RUNTIME FAIL${NC}"
            FAILED_TESTS+=("$test_name")
            ((FAIL++)) || true
        fi
    else
        echo -e "${RED}❌ COMPILE FAIL${NC}"
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
echo ""

if [ $FAIL -gt 0 ]; then
    echo "Failed Tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo ""
fi

TOTAL=$(( PASS + FAIL ))
if [ $TOTAL -gt 0 ]; then
    PASS_RATE=$(( PASS * 100 / TOTAL ))
    echo "Pass Rate: ${PASS_RATE}%"
fi

echo ""
rm -f "$ESHKOL_TEST_OUT" "$ESHKOL_TEST_BIN"

if [ $FAIL -eq 0 ]; then
    exit 0
else
    exit 1
fi
