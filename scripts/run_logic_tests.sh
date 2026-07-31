#!/bin/bash
# Run Eshkol Consciousness Engine tests — v1.1-accelerate
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
eshkol_test_isolation_init "logic"

# Honour $BUILD_DIR (CI passes it via the matrix); fall back to "build" for plain local runs.
BUILD_DIR="${BUILD_DIR:-build}"


ESHKOL="./$BUILD_DIR/eshkol-run"
PASS=0
FAIL=0

echo "=== Eshkol Consciousness Engine Tests ==="
echo ""

for test in tests/logic/*.esk; do
    name=$(basename "$test" .esk)
    printf "  %-30s " "$name"

    # Compile
    if $ESHKOL "$test" -o "$ESHKOL_TEST_BIN" 2>/dev/null; then
        # Run. A zero exit status is not a pass — the output was captured all
        # along but never inspected, so printed FAIL lines were scored PASS.
        if "$ESHKOL_TEST_BIN" >"$ESHKOL_TEST_OUT" 2>&1; then
            if eshkol_test_output_has_failure "$ESHKOL_TEST_OUT" 'error:'; then
                echo "FAIL (assertion)"
                eshkol_test_output_failures "$ESHKOL_TEST_OUT" 'error:' 10 | sed 's/^/    /'
                FAIL=$((FAIL + 1))
            else
                echo "PASS"
                PASS=$((PASS + 1))
            fi
        else
            echo "FAIL (runtime error)"
            cat "$ESHKOL_TEST_OUT"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "FAIL (compile error)"
        $ESHKOL "$test" -o "$ESHKOL_TEST_BIN" 2>&1 | head -20
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "Results: $PASS passed, $FAIL failed"
exit $FAIL
