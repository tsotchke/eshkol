#!/bin/bash
#
# core.dbsp acceptance gate (ADR 0009, v1.5.0 incremental-dataflow slice).
#
# Runs tests/stdlib/dbsp_test.esk under BOTH execution modes:
#   1. JIT  : eshkol-run -r <test>
#   2. AOT  : eshkol-run -o <bin> <test> && <bin>
#
# The test is self-asserting (prints PASS/FAIL per case, exits non-zero on any
# failure). This script fails if either mode fails to build or run clean.
#
# Honours $BUILD_DIR (CI passes it via the matrix); falls back to "build".

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

BUILD_DIR="${BUILD_DIR:-build}"
RUN="$BUILD_DIR/eshkol-run"
TEST="tests/stdlib/dbsp_test.esk"

# Resolve (require core.dbsp) against THIS checkout's lib/, which matters when
# the eshkol-run binary lives in a different checkout (e.g. a git worktree).
export ESHKOL_PATH="$PROJECT_ROOT/lib${ESHKOL_PATH:+:$ESHKOL_PATH}"

if [ ! -x "$RUN" ]; then
    echo "ERROR: $RUN not found or not executable (set BUILD_DIR?)." >&2
    exit 2
fi

FAILURES=0

# Per-run, per-repo-root isolation and the shared honest-detection helpers.
ESHKOL_TEST_ISOLATION_NO_TRAP=1
# Sourcing must be checked *before* the fact: bash 3.2 (macOS) exits the
# shell when `source` cannot find its file, so a trailing `|| {...}` never
# runs there. A suite with no prelude has no failure detection and no
# scratch isolation, and must refuse to run rather than report a PASS.
ESHKOL_TEST_LIB="$SCRIPT_DIR/lib/test_isolation.sh"
if [ ! -r "$ESHKOL_TEST_LIB" ]; then
    echo "FATAL: cannot read $ESHKOL_TEST_LIB" >&2
    echo "       (the shared test isolation and failure-detection prelude)." >&2
    echo "       Refusing to run: without it this suite would report a" >&2
    echo "       meaningless PASS." >&2
    exit 2
fi
source "$ESHKOL_TEST_LIB"
eshkol_test_isolation_init "dbsp-gate"
trap eshkol_test_isolation_cleanup EXIT

# The header above says the test "exits non-zero on any failure". Nothing
# enforced that: the output streamed straight to the terminal and the verdict
# came from the exit status alone, so a run that printed FAIL lines and exited 0
# was reported as "JIT: PASS". Capture the output and read it.
DBSP_OUT="$ESHKOL_TEST_TMPDIR/dbsp.out"

# $1 — label for the message; runs the rest as the command under test.
dbsp_check_output() {
    local label="$1"
    if eshkol_test_output_has_failure "$DBSP_OUT"; then
        echo "$label: FAIL (test reported failures while exiting 0)"
        eshkol_test_output_failures "$DBSP_OUT" "" 10 | sed 's/^/  /'
        return 1
    fi
    if eshkol_test_output_is_silent "$DBSP_OUT"; then
        echo "$label: FAIL (no output — absence of a verdict is not a pass)"
        return 1
    fi
    return 0
}

echo "========================================="
echo "  core.dbsp acceptance gate (ADR 0009)"
echo "========================================="

echo ""
echo "--- [1/2] JIT (-r) -----------------------"
if "$RUN" -r "$TEST" > "$DBSP_OUT" 2>&1; then
    cat "$DBSP_OUT"
    if dbsp_check_output "JIT"; then
        echo "JIT: PASS"
    else
        FAILURES=$((FAILURES + 1))
    fi
else
    cat "$DBSP_OUT"
    echo "JIT: FAIL"
    FAILURES=$((FAILURES + 1))
fi

echo ""
echo "--- [2/2] AOT (compile + run) ------------"
AOT_BIN="$ESHKOL_TEST_TMPDIR/dbsp_aot"
if "$RUN" -o "$AOT_BIN" "$TEST"; then
    if "$AOT_BIN" > "$DBSP_OUT" 2>&1; then
        cat "$DBSP_OUT"
        if dbsp_check_output "AOT"; then
            echo "AOT: PASS"
        else
            FAILURES=$((FAILURES + 1))
        fi
    else
        cat "$DBSP_OUT"
        echo "AOT: FAIL (runtime)"
        FAILURES=$((FAILURES + 1))
    fi
else
    echo "AOT: FAIL (compile)"
    FAILURES=$((FAILURES + 1))
fi
rm -f -- "${AOT_BIN:?}"

echo ""
echo "========================================="
if [ "$FAILURES" -eq 0 ]; then
    echo "core.dbsp gate: PASS (JIT + AOT)"
    exit 0
else
    echo "core.dbsp gate: FAIL ($FAILURES mode(s) failed)"
    exit 1
fi
