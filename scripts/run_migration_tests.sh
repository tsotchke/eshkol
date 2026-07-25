#!/bin/bash

# Eshkol Migration Test Suite
# Validates type system migration and pointer consolidation

set -e

# Per-run, per-repo-root isolation for temp files and build artifacts.
# Two suites (two worktrees, two agents, CI plus a local run) must never share
# a scratch path or a build artifact — see scripts/lib/test_isolation.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/test_isolation.sh"
eshkol_test_isolation_init "migration"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Honour $BUILD_DIR (CI passes it via the matrix); fall back to "build" for plain local runs.
BUILD_DIR="${BUILD_DIR:-build}"

# Counters
PASS=0
FAIL=0
COMPILE_FAIL=0

declare -a FAILED_TESTS

echo "========================================="
echo "  Eshkol Migration Test Suite"
echo "========================================="
echo ""

if [ ! -f "$BUILD_DIR/eshkol-run" ]; then
    echo -e "${RED}Error: eshkol-run not found. Run make first.${NC}"
    exit 1
fi

for test_file in tests/migration/*.esk; do
    [ -f "$test_file" ] || continue
    test_name=$(basename "$test_file")
    printf "Testing %-50s " "$test_name"

    eshkol_test_reset_bin
    if ./$BUILD_DIR/eshkol-run -L./$BUILD_DIR "$test_file" -o "$ESHKOL_TEST_BIN" > /dev/null 2>&1; then
        if "$ESHKOL_TEST_BIN" > "$ESHKOL_TEST_OUT" 2>&1; then
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
echo ""

if [ $FAIL -gt 0 ]; then
    echo "Failed Tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
fi

echo ""
rm -f "$ESHKOL_TEST_OUT" "$ESHKOL_TEST_BIN"

if [ $FAIL -eq 0 ]; then
    exit 0
else
    exit 1
fi
