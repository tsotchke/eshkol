#!/bin/bash

# Eshkol Examples Test Suite
# Tests all examples and categorizes them by status
# Helps identify which examples to keep, modify, or remove

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Counters
PASS=0
COMPILE_FAIL=0
RUNTIME_FAIL=0
RUNTIME_ERROR=0

# Results arrays
declare -a WORKING_EXAMPLES
declare -a COMPILE_FAILURES
declare -a RUNTIME_FAILURES
declare -a RUNTIME_ERRORS

print_examples_banner() {
    echo "========================================="
    echo "  Eshkol Examples Test Suite"
    echo "========================================="
    echo ""
}

# Honour $BUILD_DIR (CI passes it via the matrix: build / build-xla /
# build-cuda / build-asan); fall back to "build" for plain local runs.
BUILD_DIR="${BUILD_DIR:-build}"

cleanup_example_artifacts() {
    rm -f a.out a.out.tmp.o
}

run_log_has_runtime_error() {
    local run_log="$1"

    # Match actual diagnostics without flagging ordinary domain text such as
    # "absolute error" or "squared error" in numerical examples.
    grep -Eiq '^[[:space:]]*(error|fatal error|runtime error|panic|exception|uncaught|unhandled|assertion failed|segmentation fault|abort(ed)?)([[:space:]:]|$)|(^|[^[:alnum:]_])(segmentation fault|abort trap|core dumped)([^[:alnum:]_]|$)' "$run_log"
}

ensure_examples_build() {
    if [ ! -d "$BUILD_DIR" ]; then
        echo -e "${RED}Error: build directory '$BUILD_DIR' not found. Run cmake first.${NC}"
        exit 1
    fi

    if [ ! -f "$BUILD_DIR/eshkol-run" ]; then
        echo -e "${RED}Error: eshkol-run not found in '$BUILD_DIR'. Run make first.${NC}"
        exit 1
    fi

    if [ ! -f "$BUILD_DIR/stdlib.o" ]; then
        echo -e "${YELLOW}Warning: stdlib.o not found in $BUILD_DIR. Building...${NC}"
        cmake --build "$BUILD_DIR" --target stdlib
    fi
}

example_should_skip() {
    local test_name="$1"

    # Quantum-chemistry examples require the Moonlab backend; skip them unless
    # the quantum lane is enabled (mirrors the ESHKOL_QUANTUM_ENABLED gating used
    # by the quantum test suite, e.g. run_ad_adversarial.sh).
    case "$test_name" in
        vqe_h2.esk|h2_vibrational_quantum.esk|h2_vibrational_full.esk|qng_vqe.esk)
            if [ "${ESHKOL_QUANTUM_ENABLED:-OFF}" != "ON" ]; then
                return 0
            fi
            ;;
    esac

    case "$test_name" in
        selene_*|qllm_*|agent.esk|consciousness_*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

print_empty_examples_summary() {
    echo "No examples found in examples/ directory. Skipping."
    echo ""
    echo "========================================="
    echo "  Test Results Summary"
    echo "========================================="
    echo "Total Examples:     0"
    echo "Pass Rate: N/A (no examples to test)"
    echo ""
}

# Output directory for logs
LOG_DIR="/tmp/eshkol_examples_test"
mkdir -p "$LOG_DIR"

print_examples_banner
ensure_examples_build

echo "Testing all .esk files in examples/ directory..."
echo "Log files will be saved to: $LOG_DIR"
echo ""

# Check if examples directory exists and has .esk files
if [ ! -d "examples" ] || [ -z "$(ls -A examples/*.esk 2>/dev/null)" ]; then
    print_empty_examples_summary
    cleanup_example_artifacts
    exit 0
fi

# Count total examples
TOTAL=$(ls -1 examples/*.esk 2>/dev/null | wc -l | tr -d ' ')
CURRENT=0

# Run each test
for test_file in examples/*.esk; do
    ((++CURRENT))
    test_name=$(basename "$test_file")

    # Skip proprietary/unreleased examples
    if example_should_skip "$test_name"; then
        printf "SKIP (proprietary)\n"
        continue
    fi
    printf "[%3d/%3d] %-50s " "$CURRENT" "$TOTAL" "$test_name"

    # Clean up stale temp files before each test
    cleanup_example_artifacts

    # Log file for this example
    compile_log="$LOG_DIR/${test_name%.esk}_compile.log"
    run_log="$LOG_DIR/${test_name%.esk}_run.log"

    # Try to compile
    if ./"$BUILD_DIR"/eshkol-run -L./"$BUILD_DIR" "$test_file" > "$compile_log" 2>&1; then
        # Compilation succeeded, try to run
        if ./a.out > "$run_log" 2>&1; then
            exit_code=$?
            # Check if there were any errors in output
            if run_log_has_runtime_error "$run_log"; then
                echo -e "${YELLOW}⚠ RUNTIME ERROR${NC}"
                RUNTIME_ERRORS+=("$test_name")
                ((RUNTIME_ERROR++))
            else
                echo -e "${GREEN}✅ PASS${NC}"
                WORKING_EXAMPLES+=("$test_name")
                ((PASS++)) || true
            fi
        else
            exit_code=$?
            if [ $exit_code -eq 139 ]; then
                echo -e "${RED}❌ SEGFAULT${NC}"
                RUNTIME_FAILURES+=("$test_name (segfault)")
            else
                echo -e "${RED}❌ RUNTIME FAIL (exit $exit_code)${NC}"
                RUNTIME_FAILURES+=("$test_name (exit $exit_code)")
            fi
            ((RUNTIME_FAIL++))
        fi
    else
        echo -e "${RED}❌ COMPILE FAIL${NC}"
        COMPILE_FAILURES+=("$test_name")
        ((COMPILE_FAIL++)) || true
    fi
done

echo ""
echo "========================================="
echo "  Test Results Summary"
echo "========================================="
TOTAL_TESTS=$(( PASS + COMPILE_FAIL + RUNTIME_FAIL + RUNTIME_ERROR ))
echo -e "Total Examples:     $TOTAL_TESTS"
echo -e "${GREEN}Working:            $PASS${NC}"
echo -e "${RED}Compile Failures:   $COMPILE_FAIL${NC}"
echo -e "${RED}Runtime Failures:   $RUNTIME_FAIL${NC}"
echo -e "${YELLOW}Runtime Errors:     $RUNTIME_ERROR${NC}"
echo ""

# Calculate pass rate
if [ $TOTAL_TESTS -gt 0 ]; then
    PASS_RATE=$(( PASS * 100 / TOTAL_TESTS ))
    echo "Pass Rate: ${PASS_RATE}%"
    echo ""
fi

# Report working examples
if [ ${#WORKING_EXAMPLES[@]} -gt 0 ]; then
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}  Working Examples (KEEP)${NC}"
    echo -e "${GREEN}=========================================${NC}"
    for example in "${WORKING_EXAMPLES[@]}"; do
        echo "  ✅ $example"
    done
    echo ""
fi

# Report compile failures
if [ ${#COMPILE_FAILURES[@]} -gt 0 ]; then
    echo -e "${RED}=========================================${NC}"
    echo -e "${RED}  Compile Failures (REVIEW/FIX)${NC}"
    echo -e "${RED}=========================================${NC}"
    for example in "${COMPILE_FAILURES[@]}"; do
        echo "  ❌ $example"
        # Show first few lines of error
        log_file="$LOG_DIR/${example%.esk}_compile.log"
        if [ -f "$log_file" ]; then
            echo "     Error: $(grep -m1 'error:' "$log_file" 2>/dev/null || head -1 "$log_file")"
        fi
    done
    echo ""
fi

# Report runtime failures
if [ ${#RUNTIME_FAILURES[@]} -gt 0 ]; then
    echo -e "${RED}=========================================${NC}"
    echo -e "${RED}  Runtime Failures (REVIEW/FIX)${NC}"
    echo -e "${RED}=========================================${NC}"
    for example in "${RUNTIME_FAILURES[@]}"; do
        echo "  ❌ $example"
    done
    echo ""
fi

# Report runtime errors
if [ ${#RUNTIME_ERRORS[@]} -gt 0 ]; then
    echo -e "${YELLOW}=========================================${NC}"
    echo -e "${YELLOW}  Runtime Errors (REVIEW)${NC}"
    echo -e "${YELLOW}=========================================${NC}"
    for example in "${RUNTIME_ERRORS[@]}"; do
        echo "  ⚠ $example"
    done
    echo ""
fi

echo "========================================="
echo "  Log files saved to: $LOG_DIR"
echo "========================================="
echo ""
echo "To view compile errors:  cat $LOG_DIR/<example>_compile.log"
echo "To view runtime output:  cat $LOG_DIR/<example>_run.log"
echo ""

# Clean up
cleanup_example_artifacts

# Exit with appropriate code
if [ $COMPILE_FAIL -eq 0 ] && [ $RUNTIME_FAIL -eq 0 ]; then
    exit 0
else
    exit 1
fi
