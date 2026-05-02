#!/bin/bash

# Eshkol TCO (Tail Call Optimization) Test Suite
# Tests that TCO works correctly with nested letrec expressions.
# These patterns previously caused stack overflow due to TCO context corruption.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Portable timeout wrapper (macOS has no timeout command)
run_with_timeout() {
    local timeout_secs=$1
    shift
    if command -v timeout &>/dev/null; then
        timeout "$timeout_secs" "$@"
    elif command -v gtimeout &>/dev/null; then
        gtimeout "$timeout_secs" "$@"
    else
        "$@" &
        local pid=$!
        ( sleep "$timeout_secs"; kill -9 $pid 2>/dev/null ) &
        local watcher=$!
        wait $pid 2>/dev/null
        local exit_code=$?
        kill $watcher 2>/dev/null 2>&1
        wait $watcher 2>/dev/null 2>&1
        return $exit_code
    fi
}

# Counters
PASS=0
FAIL=0

# Results arrays
declare -a FAILED_TESTS

echo "========================================="
echo "  Eshkol TCO (Tail Call Optimization) Tests"
echo "========================================="
echo ""

# Honour $BUILD_DIR (CI passes it via the matrix: build / build-xla /
# build-cuda / build-asan); fall back to "build" for plain local runs.
BUILD_DIR="${BUILD_DIR:-build}"

if [ ! -d "$BUILD_DIR" ] || [ ! -f "$BUILD_DIR/eshkol-run" ]; then
    echo -e "${RED}Error: Build directory not found or eshkol-run missing.${NC}"
    echo "Please build first: cd build && cmake .. && make -j8"
    exit 1
fi

echo -e "${GREEN}Using build directory: $BUILD_DIR${NC}"
echo ""

# Check for test directory
TCO_TEST_DIR="tests/tco"

if [ ! -d "$TCO_TEST_DIR" ]; then
    echo -e "${RED}Error: $TCO_TEST_DIR directory not found.${NC}"
    exit 1
fi

# Count test files
TEST_COUNT=$(ls "$TCO_TEST_DIR"/*.esk 2>/dev/null | wc -l | tr -d ' ')
if [ "$TEST_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}No TCO test files found (*.esk).${NC}"
    exit 0
fi

echo "Found $TEST_COUNT test file(s)"
echo ""

# Run each test file
for test_file in "$TCO_TEST_DIR"/*.esk; do
    test_name=$(basename "$test_file" .esk)
    echo -n "  $test_name ... "

    # Compile
    TEMP_BIN=$(mktemp /tmp/tco_test_XXXXXX)
    set +e
    COMPILE_OUTPUT=$(./$BUILD_DIR/eshkol-run "$test_file" -o "$TEMP_BIN" 2>&1)
    COMPILE_EXIT=$?
    set -e

    if [ $COMPILE_EXIT -ne 0 ]; then
        echo -e "${RED}COMPILE FAIL${NC}"
        echo "    Compile output: $COMPILE_OUTPUT"
        ((FAIL++)) || true
        FAILED_TESTS+=("$test_name (compile)")
        rm -f "$TEMP_BIN"
        continue
    fi

    # The 3-level-nesting test in nested_tco_test.esk runs (outer 4000)
    # which the comment in the test explains intentionally trades depth
    # for per-frame size — each nested-letrec frame on Linux x64 is
    # ~80 KB (-O0; closure env + intermediate result allocas), so 4000
    # iterations need ~320 MB of stack.  Linux's default ulimit -s of
    # 8192 KB (8 MB) caps the process at ~100 frames before SIGSEGV.
    # The eshkol-run linker flag `-Wl,-z,stack-size=536870912` and the
    # runtime `eshkol_init_stack_size()` setrlimit call only affect
    # newly-created stacks (mmap'd thread stacks); the kernel sets the
    # main-thread stack size from RLIMIT_STACK at exec(), and post-exec
    # setrlimit cannot grow the main stack.  Raise the soft limit
    # before invoking each test so children inherit a 512 MB stack.
    ulimit -s 524288 2>/dev/null || ulimit -s unlimited 2>/dev/null || true

    # Run with timeout (TCO bugs cause infinite recursion → stack overflow)
    RUN_OUTPUT=$(run_with_timeout 60 "$TEMP_BIN" 2>&1)
    RUN_EXIT=$?

    rm -f "$TEMP_BIN"

    if [ $RUN_EXIT -ne 0 ]; then
        echo -e "${RED}RUNTIME FAIL (exit $RUN_EXIT)${NC}"
        echo "    Output: $RUN_OUTPUT"
        ((FAIL++)) || true
        FAILED_TESTS+=("$test_name (runtime exit $RUN_EXIT)")
        continue
    fi

    # Check for FAIL in output
    if echo "$RUN_OUTPUT" | grep -q "FAIL"; then
        echo -e "${RED}FAIL${NC}"
        echo "    Output: $RUN_OUTPUT"
        ((FAIL++)) || true
        FAILED_TESTS+=("$test_name (wrong result)")
        continue
    fi

    # Check for PASS in output
    if echo "$RUN_OUTPUT" | grep -q "PASS"; then
        echo -e "${GREEN}PASS${NC}"
        ((PASS++)) || true
    else
        echo -e "${YELLOW}UNKNOWN${NC}"
        echo "    Output: $RUN_OUTPUT"
        ((FAIL++)) || true
        FAILED_TESTS+=("$test_name (no PASS/FAIL)")
    fi
done

echo ""
echo "========================================="
echo "  TCO Test Results"
echo "========================================="
echo "Passed: $PASS"
echo "Failed: $FAIL"

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed tests:${NC}"
    for t in "${FAILED_TESTS[@]}"; do
        echo "  - $t"
    done
fi

echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All TCO tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some TCO tests failed.${NC}"
    exit 1
fi
