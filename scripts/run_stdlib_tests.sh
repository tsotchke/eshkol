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

detect_jobs() {
    if [ -n "${ESHKOL_SUITE_JOBS:-}" ]; then
        echo "$ESHKOL_SUITE_JOBS"
        return
    fi
    local cpus=1
    if command -v nproc >/dev/null 2>&1; then
        cpus=$(nproc 2>/dev/null || echo 1)
    elif command -v sysctl >/dev/null 2>&1; then
        cpus=$(sysctl -n hw.ncpu 2>/dev/null || echo 1)
    fi
    if [ "$cpus" -gt 3 ]; then
        cpus=3
    elif [ "$cpus" -lt 1 ]; then
        cpus=1
    fi
    echo "$cpus"
}

MAX_JOBS=$(detect_jobs)
echo "Parallel workers: $MAX_JOBS"
echo ""

declare -a TEST_FILES
declare -a TEST_NAMES
TEST_COUNT=0
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
    TEST_FILES+=("$test_file")
    TEST_NAMES+=("$test_name")
    ((TEST_COUNT++)) || true
done

run_one_stdlib_test() {
    local idx="$1"
    local test_file="$2"
    local test_name="$3"
    local test_bin="$RUN_DIR/${test_name%.esk}"
    local test_output="$RUN_DIR/${test_name%.esk}.out"
    local compile_output="$RUN_DIR/${test_name%.esk}.compile.out"
    local status_file="$RUN_DIR/${idx}.status"
    local worker_tmp="$RUN_DIR/tmp_${idx}"
    mkdir -p "$worker_tmp"

    rm -f "$test_bin" "$test_bin.tmp.o" "$test_output" "$compile_output" "$status_file"

    if TMPDIR="$worker_tmp" "$ESHKOL_RUN" -I "$ROOT_DIR/tests" -L"$ROOT_DIR/$BUILD_DIR" "$test_file" -o "$test_bin" > "$compile_output" 2>&1; then
        if TMPDIR="$worker_tmp" "$test_bin" > "$test_output" 2>&1; then
            if [ "$test_name" = "test_condition_counts.esk" ]; then
                if grep -Fxq "PASS: condition counts" "$test_output"; then
                    echo "PASS" > "$status_file"
                    printf "Testing %-50s ${GREEN}✅ PASS${NC}\n" "$test_name"
                else
                    echo "RUNTIME_ERROR" > "$status_file"
                    printf "Testing %-50s ${YELLOW}⚠ RUNTIME ERROR${NC}\n" "$test_name"
                fi
            elif eshkol_test_output_has_failure "$test_output" 'error:'; then
                echo "RUNTIME_ERROR" > "$status_file"
                printf "Testing %-50s ${YELLOW}⚠ RUNTIME ERROR${NC}\n" "$test_name"
            else
                echo "PASS" > "$status_file"
                printf "Testing %-50s ${GREEN}✅ PASS${NC}\n" "$test_name"
            fi
        else
            echo "RUNTIME_FAIL" > "$status_file"
            printf "Testing %-50s ${RED}❌ RUNTIME FAIL${NC}\n" "$test_name"
        fi
    else
        echo "COMPILE_FAIL" > "$status_file"
        printf "Testing %-50s ${RED}❌ COMPILE FAIL${NC}\n" "$test_name"
    fi
    rm -f "$test_bin" "$test_bin.tmp.o"
    rm -rf "$worker_tmp"
}

declare -a ACTIVE_PIDS
ACTIVE_PIDS=()

wait_for_stdlib_slot() {
    while [ "${#ACTIVE_PIDS[@]}" -ge "$MAX_JOBS" ]; do
        local next_pids=()
        local p
        for p in "${ACTIVE_PIDS[@]}"; do
            if kill -0 "$p" 2>/dev/null; then
                next_pids+=("$p")
            else
                wait "$p" || true
            fi
        done
        if [ "${#next_pids[@]}" -gt 0 ]; then
            ACTIVE_PIDS=("${next_pids[@]}")
        else
            ACTIVE_PIDS=()
        fi
        if [ "${#ACTIVE_PIDS[@]}" -ge "$MAX_JOBS" ]; then
            sleep 0.2
        fi
    done
}

for (( idx=0; idx<TEST_COUNT; idx++ )); do
    if [ "$MAX_JOBS" -le 1 ]; then
        run_one_stdlib_test "$idx" "${TEST_FILES[$idx]}" "${TEST_NAMES[$idx]}"
    else
        wait_for_stdlib_slot
        ( run_one_stdlib_test "$idx" "${TEST_FILES[$idx]}" "${TEST_NAMES[$idx]}" ) &
        ACTIVE_PIDS+=("$!")
    fi
done

if [ "${#ACTIVE_PIDS[@]}" -gt 0 ]; then
    for p in "${ACTIVE_PIDS[@]}"; do
        wait "$p" || true
    done
fi

for (( idx=0; idx<TEST_COUNT; idx++ )); do
    test_name="${TEST_NAMES[$idx]}"
    status_file="$RUN_DIR/${idx}.status"
    test_output="$RUN_DIR/${test_name%.esk}.out"
    compile_output="$RUN_DIR/${test_name%.esk}.compile.out"
    verdict="RUNTIME_FAIL"
    if [ -r "$status_file" ]; then
        verdict=$(cat "$status_file")
    fi
    case "$verdict" in
        PASS)
            ((PASS++)) || true
            ;;
        RUNTIME_ERROR)
            echo "  [detail] $test_name (RUNTIME ERROR):"
            head -n "$FAILURE_LINES" "$test_output" 2>/dev/null | sed 's/^/    /'
            RUNTIME_ERRORS+=("$test_name")
            ((FAIL++)) || true
            ;;
        RUNTIME_FAIL)
            echo "  [detail] $test_name (RUNTIME FAIL):"
            head -n "$FAILURE_LINES" "$test_output" 2>/dev/null | sed 's/^/    /'
            FAILED_TESTS+=("$test_name")
            ((FAIL++)) || true
            ;;
        COMPILE_FAIL)
            echo "  [detail] $test_name (COMPILE FAIL):"
            head -n "$FAILURE_LINES" "$compile_output" 2>/dev/null | sed 's/^/    /'
            FAILED_TESTS+=("$test_name")
            ((COMPILE_FAIL++)) || true
            ((FAIL++)) || true
            ;;
    esac
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
