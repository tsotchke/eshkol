#!/bin/bash

# Eshkol XLA/StableHLO Integration Test Suite
# Tests XLA backend dispatch and tensor operations
#
# Prerequisites:
# - Build with: cmake -DESHKOL_USE_XLA=ON -B build-xla && cmake --build build-xla
# - Or lite build: cmake -B build && cmake --build build

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASS=0
FAIL=0
SKIP=0
COMPILE_FAIL=0

# Results arrays
declare -a FAILED_TESTS
declare -a SKIPPED_TESTS
declare -a RUNTIME_ERRORS

echo "========================================="
echo "  Eshkol XLA/StableHLO Integration Tests"
echo "========================================="
echo ""

# Determine which build directory to use
# Override with: BUILD_DIR=build ./scripts/run_xla_tests.sh
XLA_ENABLED="no"

if [ -n "$BUILD_DIR" ]; then
    # User-specified build directory
    if [ ! -d "$BUILD_DIR" ] || [ ! -f "$BUILD_DIR/eshkol-run" ]; then
        echo -e "${RED}Error: Specified BUILD_DIR=$BUILD_DIR not found or missing eshkol-run.${NC}"
        exit 1
    fi
    if [ -f "$BUILD_DIR/CMakeCache.txt" ] && \
       grep -qE "ESHKOL_(USE_)?XLA(_ENABLED)?:BOOL=ON" "$BUILD_DIR/CMakeCache.txt"; then
        XLA_ENABLED="yes"
    fi
    echo -e "${GREEN}Using build directory: $BUILD_DIR${NC}"
else
    # Auto-detect: prefer build-xla, fall back to build
    for candidate in build-xla build; do
        if [ -d "$candidate" ] && [ -f "$candidate/eshkol-run" ]; then
            BUILD_DIR="$candidate"
            if [ -f "$candidate/CMakeCache.txt" ] && \
               grep -qE "ESHKOL_(USE_)?XLA(_ENABLED)?:BOOL=ON" "$candidate/CMakeCache.txt"; then
                XLA_ENABLED="yes"
            fi
            break
        fi
    done
    if [ -z "$BUILD_DIR" ]; then
        echo -e "${RED}Error: No build directory found.${NC}"
        echo "Please build with one of:"
        echo "  cmake -DESHKOL_USE_XLA=ON -B build-xla && cmake --build build-xla"
        echo "  cmake -B build && cmake --build build"
        echo "Or specify: BUILD_DIR=<path> $0"
        exit 1
    fi
    echo -e "${GREEN}Using build directory: $BUILD_DIR (auto-detected)${NC}"
fi

if [ "$XLA_ENABLED" = "yes" ]; then
    echo -e "${GREEN}XLA support: ENABLED${NC}"
else
    echo -e "${CYAN}Note: XLA tests will use BLAS/SIMD fallback.${NC}"
fi

echo ""

# ===== Part 1: C++ Unit Tests =====
echo -e "${BLUE}===== C++ Unit Tests =====${NC}"

XLA_TEST_BIN="$BUILD_DIR/xla_codegen_test"

if [ -f "$XLA_TEST_BIN" ]; then
    echo "Running XLA C++ unit tests..."
    if $XLA_TEST_BIN; then
        echo -e "${GREEN}✅ C++ Unit Tests: PASS${NC}"
        ((PASS++)) || true
    else
        echo -e "${RED}❌ C++ Unit Tests: FAIL${NC}"
        FAILED_TESTS+=("C++ Unit Tests")
        ((FAIL++)) || true
    fi
else
    echo -e "${YELLOW}⚠ C++ Unit Tests: SKIPPED (test binary not built)${NC}"
    echo "  To enable: add xla_codegen_test target to CMakeLists.txt"
    SKIPPED_TESTS+=("C++ Unit Tests")
    ((SKIP++)) || true
fi

echo ""

# ===== Part 2: Eshkol Language Tests =====
echo -e "${BLUE}===== Eshkol Language Tests =====${NC}"

XLA_TEST_DIR="tests/xla"

# Check if test directory exists
if [ ! -d "$XLA_TEST_DIR" ]; then
    echo -e "${YELLOW}Warning: $XLA_TEST_DIR directory not found. Creating...${NC}"
    mkdir -p "$XLA_TEST_DIR"
fi

# Count test files
TEST_COUNT=$(find "$XLA_TEST_DIR" -name "*.esk" 2>/dev/null | wc -l | tr -d ' ')
if [ "$TEST_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}No XLA test files found (*.esk).${NC}"
else
    echo "Found $TEST_COUNT test file(s)"
    echo ""

    # Run each .esk test
    for test_file in "$XLA_TEST_DIR"/*.esk; do
        if [ ! -f "$test_file" ]; then
            continue
        fi

        test_name=$(basename "$test_file")
        printf "Testing %-45s " "$test_name"

        # Clean up stale temp files
        rm -f a.out a.out.tmp.o

        # Try to compile
        if ./$BUILD_DIR/eshkol-run "$test_file" -L./$BUILD_DIR > /tmp/xla_compile.log 2>&1; then
            # Compilation succeeded, try to run
            if ./a.out > /tmp/xla_test_output.txt 2>&1; then
                # Check for FAIL markers in output (from test assertions)
                if grep -q "^FAIL:" /tmp/xla_test_output.txt; then
                    echo -e "${RED}❌ ASSERTION FAIL${NC}"
                    FAILED_TESTS+=("$test_name")
                    ((FAIL++)) || true
                    # Show failed assertions
                    grep -i "fail" /tmp/xla_test_output.txt | head -3 | sed 's/^/    /'
                elif grep -v "PASS" /tmp/xla_test_output.txt | grep -qi "error:"; then
                    echo -e "${YELLOW}⚠ RUNTIME ERROR${NC}"
                    RUNTIME_ERRORS+=("$test_name")
                    ((FAIL++)) || true
                else
                    echo -e "${GREEN}✅ PASS${NC}"
                    ((PASS++)) || true
                fi
            else
                exit_code=$?
                echo -e "${RED}❌ RUNTIME FAIL (exit $exit_code)${NC}"
                FAILED_TESTS+=("$test_name")
                ((FAIL++)) || true
            fi
        else
            echo -e "${RED}❌ COMPILE FAIL${NC}"
            FAILED_TESTS+=("$test_name")
            ((COMPILE_FAIL++)) || true
            ((FAIL++)) || true
            # Show compile error (last few lines)
            tail -3 /tmp/xla_compile.log 2>/dev/null | sed 's/^/    /'
        fi
    done
fi

echo ""

# ===== Part 3: Performance Sanity Check =====
echo -e "${BLUE}===== Performance Sanity Check =====${NC}"

# Quick benchmark to verify matmul path is working
cat > /tmp/xla_perf_test.esk << 'ESKEOF'
;; Quick performance sanity check
(define N 150)
(define A (reshape (arange (* N N)) N N))
(define B (reshape (ones (* N N)) N N))
(define C (matmul A B))
(display "150x150 matmul: ")
(display (tensor-shape C))
(newline)
ESKEOF

printf "Testing %-45s " "performance_sanity"

if ./$BUILD_DIR/eshkol-run /tmp/xla_perf_test.esk -L./$BUILD_DIR > /tmp/xla_compile.log 2>&1; then
    start_time=$(python3 -c "import time; print(int(time.time() * 1000))" 2>/dev/null || date +%s%3N)
    TIMEOUT_CMD="timeout"
    if ! command -v timeout &>/dev/null; then
        if command -v gtimeout &>/dev/null; then
            TIMEOUT_CMD="gtimeout"
        else
            TIMEOUT_CMD=""
        fi
    fi
    if [ -n "$TIMEOUT_CMD" ]; then
        PERF_RUN="$TIMEOUT_CMD 60 ./a.out"
    else
        PERF_RUN="./a.out"
    fi
    if $PERF_RUN > /tmp/xla_perf_output.txt 2>&1; then
        end_time=$(python3 -c "import time; print(int(time.time() * 1000))" 2>/dev/null || date +%s%3N)
        elapsed=$((end_time - start_time))
        echo -e "${GREEN}✅ PASS${NC} (${elapsed}ms)"
        ((PASS++)) || true
    else
        echo -e "${RED}❌ FAIL (timeout or error)${NC}"
        FAILED_TESTS+=("performance_sanity")
        ((FAIL++)) || true
    fi
else
    echo -e "${RED}❌ COMPILE FAIL${NC}"
    FAILED_TESTS+=("performance_sanity")
    ((COMPILE_FAIL++)) || true
    ((FAIL++)) || true
fi

rm -f /tmp/xla_perf_test.esk

echo ""

# ===== Summary =====
echo "========================================="
echo "  XLA Test Results Summary"
echo "========================================="
TOTAL=$((PASS + FAIL + SKIP))
echo "Total Tests:    $TOTAL"
echo -e "${GREEN}Passed:         $PASS${NC}"
echo -e "${RED}Failed:         $FAIL${NC}"
if [ $SKIP -gt 0 ]; then
    echo -e "${YELLOW}Skipped:        $SKIP${NC}"
fi
echo "  Compile Failures: $COMPILE_FAIL"
echo "  Runtime Errors:   ${#RUNTIME_ERRORS[@]}"
echo ""

if [ $FAIL -gt 0 ]; then
    echo "Failed Tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo ""
fi

if [ ${#RUNTIME_ERRORS[@]} -gt 0 ]; then
    echo "Runtime Errors:"
    for test in "${RUNTIME_ERRORS[@]}"; do
        echo "  - $test"
    done
    echo ""
fi

if [ $SKIP -gt 0 ]; then
    echo "Skipped Tests:"
    for test in "${SKIPPED_TESTS[@]}"; do
        echo "  - $test"
    done
    echo ""
fi

# Calculate pass rate
if [ $TOTAL -gt 0 ]; then
    PASS_RATE=$(( PASS * 100 / TOTAL ))
    echo "Pass Rate: ${PASS_RATE}%"
fi

echo ""

# XLA status reminder
if [ "$XLA_ENABLED" = "yes" ]; then
    echo -e "${GREEN}Build: XLA/StableHLO enabled${NC}"
else
    echo -e "${CYAN}Build: Lite (BLAS/SIMD fallback)${NC}"
fi

# Clean up
rm -f /tmp/xla_test_output.txt /tmp/xla_compile.log /tmp/xla_perf_output.txt a.out

# Exit with appropriate code
if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All XLA tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some XLA tests failed.${NC}"
    exit 1
fi
