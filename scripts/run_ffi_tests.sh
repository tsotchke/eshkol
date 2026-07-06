#!/bin/bash

# Eshkol FFI Test Suite
# Runs all tests/ffi/*.esk tests (extern declarations, string<->C marshalling,
# byte-vs-codepoint length, etc.)

set -e

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
echo "  Eshkol FFI Test Suite"
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

echo "Testing all files in tests/ffi/ directory (AOT)..."
echo ""

for test_file in tests/ffi/*.esk; do
    [ -f "$test_file" ] || continue
    test_name=$(basename "$test_file")
    printf "Testing %-50s " "$test_name"

    rm -f a.out a.out.tmp.o

    if ./$BUILD_DIR/eshkol-run -L./$BUILD_DIR "$test_file" > /dev/null 2>&1; then
        if run_with_timeout 10 ./a.out > /tmp/ffi_test_output.txt 2>&1; then
            if grep -q "^FAIL" /tmp/ffi_test_output.txt; then
                echo -e "${RED}❌ FAIL${NC}"
                FAILED_TESTS+=("$test_name (AOT)")
                ((FAIL++)) || true
            else
                echo -e "${GREEN}✅ PASS${NC}"
                ((PASS++)) || true
            fi
        else
            echo -e "${RED}❌ RUNTIME FAIL${NC}"
            FAILED_TESTS+=("$test_name (AOT)")
            ((FAIL++)) || true
        fi
    else
        echo -e "${RED}❌ COMPILE FAIL${NC}"
        FAILED_TESTS+=("$test_name (AOT)")
        ((COMPILE_FAIL++)) || true
        ((FAIL++)) || true
    fi
done

echo ""
echo "Testing all files in tests/ffi/ directory (-r JIT)..."
echo ""

for test_file in tests/ffi/*.esk; do
    [ -f "$test_file" ] || continue
    test_name=$(basename "$test_file")
    printf "Testing %-50s " "$test_name"

    if run_with_timeout 10 ./$BUILD_DIR/eshkol-run -L./$BUILD_DIR -r "$test_file" > /tmp/ffi_test_output.txt 2>&1; then
        if grep -q "^FAIL" /tmp/ffi_test_output.txt; then
            echo -e "${RED}❌ FAIL${NC}"
            FAILED_TESTS+=("$test_name (-r)")
            ((FAIL++)) || true
        else
            echo -e "${GREEN}✅ PASS${NC}"
            ((PASS++)) || true
        fi
    else
        echo -e "${RED}❌ RUNTIME FAIL${NC}"
        FAILED_TESTS+=("$test_name (-r)")
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
rm -f /tmp/ffi_test_output.txt a.out

if [ $FAIL -eq 0 ]; then
    exit 0
else
    exit 1
fi
