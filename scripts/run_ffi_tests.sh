#!/usr/bin/env bash

# Eshkol FFI Test Suite
# Runs all tests/ffi/*.esk tests (extern declarations, string<->C marshalling,
# byte-vs-codepoint length, etc.)

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
# shellcheck source=lib/checked_write.sh
. "$(dirname "$ESHKOL_TEST_LIB")/checked_write.sh"
eshkol_test_isolation_init "ffi"

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
    local timeout_marker="$2"
    shift
    shift

    rm -f "$timeout_marker"
    "$@" &
    local cmd_pid=$!

    (
        sleep "$seconds"
        if kill -0 "$cmd_pid" 2>/dev/null; then
            : > "$timeout_marker"
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

DIAGNOSTIC_BASE="${RUNNER_TEMP:-$ESHKOL_TEST_TMPDIR}/eshkol-test-logs"
DIAGNOSTIC_DIR="$DIAGNOSTIC_BASE/ffi-diagnostics-$(basename "$ESHKOL_TEST_TMPDIR")"
mkdir -p "$DIAGNOSTIC_DIR"

record_phase() {
    local status_path="$1" phase="$2" exit_status="$3" timeout_marker="$4"
    {
        printf 'phase=%s\n' "$phase"
        printf 'exit_status=%s\n' "$exit_status"
        if [ -s "$timeout_marker" ]; then
            printf 'timed_out=yes\n'
        else
            printf 'timed_out=no\n'
        fi
        printf 'log=%s\n' "${status_path%.status}.log"
    } > "$status_path"
}

show_failed_phase() {
    local label="$1" log_path="$2" status_path="$3"
    echo "  DIAGNOSTIC: $label"
    cat "$status_path"
    if [ -s "$log_path" ]; then
        cat "$log_path"
    else
        echo "(no stdout or stderr captured)"
    fi
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

    phase_prefix="$DIAGNOSTIC_DIR/${test_name%.esk}"
    compile_log="$phase_prefix.aot-compile.log"
    compile_status="$phase_prefix.aot-compile.status"
    : > "$phase_prefix.aot-compile.timeout"

    eshkol_test_reset_bin
    if ./$BUILD_DIR/eshkol-run -L./$BUILD_DIR "$test_file" -o "$ESHKOL_TEST_BIN" > "$compile_log" 2>&1; then
        compile_rc=0
    else
        compile_rc=$?
    fi
    record_phase "$compile_status" "AOT compile" "$compile_rc" "$phase_prefix.aot-compile.timeout"
    if [ "$compile_rc" -eq 0 ]; then
        run_log="$phase_prefix.aot-run.log"
        run_status="$phase_prefix.aot-run.status"
        run_timeout="$phase_prefix.aot-run.timeout"
        eshkol_require_output_file_path "$ESHKOL_TEST_OUT"
        if run_with_timeout 10 "$run_timeout" "$ESHKOL_TEST_BIN" > "$run_log" 2>&1; then
            run_rc=0
        else
            run_rc=$?
        fi
        record_phase "$run_status" "AOT run" "$run_rc" "$run_timeout"
        if [ "$run_rc" -eq 0 ]; then
            # A failure marker anywhere in the output fails the test — the old
            # `^FAIL`-anchored match never saw the indented `  <case>: FAIL`
            # form that most test programs actually print.
            if eshkol_test_output_has_failure "$run_log"; then
                echo -e "${RED}❌ FAIL${NC}"
                FAILED_TESTS+=("$test_name (AOT)")
                ((FAIL++)) || true
            else
                echo -e "${GREEN}✅ PASS${NC}"
                ((PASS++)) || true
            fi
        else
            if [ -s "$run_timeout" ]; then
                echo -e "${RED}❌ RUNTIME TIMEOUT (10s)${NC}"
            else
                echo -e "${RED}❌ RUNTIME FAIL (exit $run_rc)${NC}"
            fi
            show_failed_phase "$test_name AOT run" "$run_log" "$run_status"
            FAILED_TESTS+=("$test_name (AOT)")
            ((FAIL++)) || true
        fi
    else
        echo -e "${RED}❌ COMPILE FAIL (exit $compile_rc)${NC}"
        show_failed_phase "$test_name AOT compile" "$compile_log" "$compile_status"
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

    phase_prefix="$DIAGNOSTIC_DIR/${test_name%.esk}"
    jit_log="$phase_prefix.jit-orc-cache0.log"
    jit_status="$phase_prefix.jit-orc-cache0.status"
    jit_timeout="$phase_prefix.jit-orc-cache0.timeout"

    eshkol_require_output_file_path "$ESHKOL_TEST_OUT"
    if ESHKOL_JIT_CACHE=0 run_with_timeout 10 "$jit_timeout" ./$BUILD_DIR/eshkol-run -L./$BUILD_DIR -r "$test_file" > "$jit_log" 2>&1; then
        jit_rc=0
    else
        jit_rc=$?
    fi
    record_phase "$jit_status" "JIT ORC (-r, ESHKOL_JIT_CACHE=0)" "$jit_rc" "$jit_timeout"
    if [ "$jit_rc" -eq 0 ]; then
        # A failure marker anywhere in the output fails the test — the old
        # `^FAIL`-anchored match never saw the indented `  <case>: FAIL`
        # form that most test programs actually print.
        if eshkol_test_output_has_failure "$jit_log"; then
            echo -e "${RED}❌ FAIL${NC}"
            FAILED_TESTS+=("$test_name (-r)")
            ((FAIL++)) || true
        else
            echo -e "${GREEN}✅ PASS${NC}"
            ((PASS++)) || true
        fi
    else
        if [ -s "$jit_timeout" ]; then
            echo -e "${RED}❌ JIT ORC TIMEOUT (10s)${NC}"
        else
            echo -e "${RED}❌ JIT ORC FAIL (exit $jit_rc)${NC}"
        fi
        show_failed_phase "$test_name JIT ORC (-r, ESHKOL_JIT_CACHE=0)" "$jit_log" "$jit_status"
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
eshkol_checked_rm "$ESHKOL_TEST_OUT" "$ESHKOL_TEST_BIN"

if [ $FAIL -eq 0 ]; then
    exit 0
else
    exit 1
fi
