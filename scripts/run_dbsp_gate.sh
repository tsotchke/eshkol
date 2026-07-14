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

echo "========================================="
echo "  core.dbsp acceptance gate (ADR 0009)"
echo "========================================="

echo ""
echo "--- [1/2] JIT (-r) -----------------------"
if "$RUN" -r "$TEST"; then
    echo "JIT: PASS"
else
    echo "JIT: FAIL"
    FAILURES=$((FAILURES + 1))
fi

echo ""
echo "--- [2/2] AOT (compile + run) ------------"
AOT_BIN="$(mktemp -t dbsp_aot.XXXXXX)"
if "$RUN" -o "$AOT_BIN" "$TEST"; then
    if "$AOT_BIN"; then
        echo "AOT: PASS"
    else
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
