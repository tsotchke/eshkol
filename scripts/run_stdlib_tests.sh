#!/usr/bin/env bash

# Eshkol Stdlib Test Suite
# Runs all stdlib tests and reports results

set -e

# Per-run, per-repo-root isolation for temp files and build artifacts.
# Two suites (two worktrees, two agents, CI plus a local run) must never share
# a scratch path or a build artifact — see scripts/lib/test_isolation.sh.
ESHKOL_TEST_ISOLATION_NO_TRAP=1
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
eshkol_test_isolation_init "stdlib"

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

ROOT_DIR="$(pwd)"
BUILD_DIR="${BUILD_DIR:-build}"
ESHKOL_RUN="$ROOT_DIR/$BUILD_DIR/eshkol-run"
FAILURE_LINES="${ESHKOL_TEST_FAILURE_LINES:-40}"
RUN_DIR="$ESHKOL_TEST_TMPDIR/run"
mkdir -p "$RUN_DIR"
cleanup() {
    # No repo-root `rm -f "$ROOT_DIR/a.out"` here: this suite compiles to
    # $RUN_DIR, and reaching into the repo root deleted whatever a *concurrent*
    # suite had just built there — the cross-suite clobber this pass removes.
    eshkol_test_isolation_cleanup
}
trap cleanup EXIT

echo "========================================="
echo "  Eshkol Stdlib Test Suite"
echo "========================================="
echo ""

# Ensure build directory exists
if [ ! -d "$ROOT_DIR/$BUILD_DIR" ]; then
    echo -e "${RED}Error: build directory not found. Run cmake first.${NC}"
    exit 1
fi

# Check if compiler exists
if [ ! -f "$ESHKOL_RUN" ]; then
    echo -e "${RED}Error: eshkol-run not found. Run make first.${NC}"
    exit 1
fi

echo "Testing all files in tests/stdlib/ directory..."
echo ""

# Run each test
for test_file in "$ROOT_DIR"/tests/stdlib/*.esk; do
    test_name=$(basename "$test_file")
    # Three long-run effective-homology checks require 29-33 GB RAM and are owned
    # by scripts/run_effective_gate.sh (excluded from default CTest as well).
    # Opt in via ESHKOL_RUN_HEAVY_EFFECTIVE_TESTS=1 on high-memory hosts.
    if [ "${ESHKOL_RUN_HEAVY_EFFECTIVE_TESTS:-0}" != "1" ]; then
        case "$test_name" in
            effective_hypercover_test.esk|effective_hypercover_3type_test.esk|effective_pi6_s3_test.esk)
                printf "Skipping %-49s (29-33 GB; owned by run_effective_gate.sh)\n" "$test_name"
                continue
                ;;
        esac
    fi
    test_bin="$RUN_DIR/${test_name%.esk}"
    test_output="$RUN_DIR/${test_name%.esk}.out"
    compile_output="$RUN_DIR/${test_name%.esk}.compile.out"
    printf "Testing %-50s " "$test_name"

    # Clean up stale temp files before each test
    rm -f "$test_bin" "$test_bin.tmp.o" "$test_output" "$compile_output"

    # Try to compile
    if "$ESHKOL_RUN" -I "$ROOT_DIR/tests" -L"$ROOT_DIR/$BUILD_DIR" "$test_file" -o "$test_bin" > "$compile_output" 2>&1; then
        # Compilation succeeded, try to run
        if "$test_bin" > "$test_output" 2>&1; then
            # test_condition_counts.esk intentionally registers 2 failing
            # core.testing checks to assert pass/fail counts, then exits 0
            # only when it prints "PASS: condition counts".
            if [ "$test_name" = "test_condition_counts.esk" ]; then
                if grep -Fxq "PASS: condition counts" "$test_output"; then
                    echo -e "${GREEN}✅ PASS${NC}"
                    ((PASS++)) || true
                else
                    echo -e "${YELLOW}⚠ RUNTIME ERROR${NC}"
                    head -n "$FAILURE_LINES" "$test_output" | sed 's/^/    /'
                    RUNTIME_ERRORS+=("$test_name")
                    ((FAIL++)) || true
                fi
            # Check if there were any errors in output
            # `error:` alone is a compiler diagnostic, not a verdict: these
            # programs print their own FAIL lines and exit 0, so scan for
            # failure markers too — anywhere on the line, not just column 0.
            elif eshkol_test_output_has_failure "$test_output" 'error:'; then
                echo -e "${YELLOW}⚠ RUNTIME ERROR${NC}"
                head -n "$FAILURE_LINES" "$test_output" | sed 's/^/    /'
                RUNTIME_ERRORS+=("$test_name")
                ((FAIL++)) || true
            else
                echo -e "${GREEN}✅ PASS${NC}"
                ((PASS++)) || true
            fi
        else
            echo -e "${RED}❌ RUNTIME FAIL${NC}"
            head -n "$FAILURE_LINES" "$test_output" | sed 's/^/    /'
            FAILED_TESTS+=("$test_name")
            ((FAIL++)) || true
        fi
    else
        echo -e "${RED}❌ COMPILE FAIL${NC}"
        head -n "$FAILURE_LINES" "$compile_output" | sed 's/^/    /'
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

# Exit with appropriate code
if [ $FAIL -eq 0 ]; then
    exit 0
else
    exit 1
fi
