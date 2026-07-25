#!/bin/bash

# Eshkol Examples Test Suite (Verbose Output)
# Same as run_examples_tests.sh but shows compile/runtime output for debugging

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
eshkol_test_isolation_init "examples-out"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "========================================="
echo "  Eshkol Examples Test Suite (Verbose)"
echo "========================================="
echo ""

# Honour $BUILD_DIR (CI passes it via the matrix); fall back to "build" for plain local runs.
BUILD_DIR="${BUILD_DIR:-build}"

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

# Check if stdlib exists
if [ ! -f "$BUILD_DIR/stdlib.o" ]; then
    echo -e "${YELLOW}Warning: stdlib.o not found. Building...${NC}"
    cmake --build "$BUILD_DIR" --target stdlib
fi

# Allow specifying specific file or pattern as argument
if [ $# -gt 0 ]; then
    FILES="$@"
else
    FILES="examples/*.esk"
fi

echo "Testing: $FILES"
echo ""

PASS=0
FAIL=0

for test_file in $FILES; do
    if [ ! -f "$test_file" ]; then
        echo -e "${RED}File not found: $test_file${NC}"
        continue
    fi

    test_name=$(basename "$test_file")

    echo "========================================="
    echo -e "${CYAN}Testing: $test_name${NC}"
    echo "========================================="

    # Clean up stale temp files
    eshkol_test_reset_bin
    echo -e "${BLUE}[Compiling...]${NC}"

    # Try to compile (show output)
    if ./$BUILD_DIR/eshkol-run -L./$BUILD_DIR "$test_file" -o "$ESHKOL_TEST_BIN" 2>&1; then
        echo ""
        echo -e "${BLUE}[Running...]${NC}"

        # Try to run (show output)
        if "$ESHKOL_TEST_BIN" 2>&1; then
            echo ""
            echo -e "${GREEN}✅ PASS${NC}"
            ((PASS++))
        else
            exit_code=$?
            echo ""
            if [ $exit_code -eq 139 ]; then
                echo -e "${RED}❌ SEGFAULT${NC}"
            else
                echo -e "${RED}❌ RUNTIME FAIL (exit $exit_code)${NC}"
            fi
            ((FAIL++))
        fi
    else
        echo ""
        echo -e "${RED}❌ COMPILE FAIL${NC}"
        ((FAIL++))
    fi

    echo ""
done

echo "========================================="
echo "  Summary"
echo "========================================="
echo -e "${GREEN}Passed: $PASS${NC}"
echo -e "${RED}Failed: $FAIL${NC}"

# Clean up
eshkol_test_reset_bin
if [ $FAIL -eq 0 ]; then
    exit 0
else
    exit 1
fi
