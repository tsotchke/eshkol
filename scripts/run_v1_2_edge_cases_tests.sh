#!/bin/bash
# Run Eshkol v1.2 edge-case + security regression suite.
#
# This pulls every .esk file under tests/v1_2_edge_cases/ and treats it
# as an AOT compile-and-run smoke test.  Each test prints its own
# pass/fail dialogue (the suite is the testing-framework "(define-test
# ...)" / "(check-equal? ...)" idiom; check-equal? prints "FAIL: ..."
# with a non-zero exit on mismatch).
#
# Per the v1.2 release audit (notes/v1.2-audit-2026-04-30.md, blocker
# #1), this suite was previously not invoked by run_all_tests.sh.
# Adding this runner closes that gap.

set -e

ESHKOL="./build/eshkol-run"
PASS=0
FAIL=0

echo "=== Eshkol v1.2 Edge-Case + Security Suite ==="
echo ""

for test in tests/v1_2_edge_cases/*.esk; do
    name=$(basename "$test" .esk)
    printf "  %-50s " "$name"

    # Honour `;; mode: jit` markers — these tests are explicitly
    # JIT-only (e.g. they exercise eval, dynamic loads, or REPL-side
    # symbol resolution that AOT compilation can't reproduce).  Run
    # them through `eshkol-run -r` instead of compile-and-run.
    if head -1 "$test" | grep -qiE "^;;\s*mode:\s*jit"; then
        if $ESHKOL -r "$test" >/tmp/eshkol_v12_run_out 2>&1; then
            echo "PASS (JIT)"
            PASS=$((PASS + 1))
        else
            echo "FAIL (JIT)"
            head -10 /tmp/eshkol_v12_run_out | sed 's/^/    /'
            FAIL=$((FAIL + 1))
        fi
        continue
    fi

    # Compile (AOT)
    if $ESHKOL "$test" -o /tmp/eshkol_v12_test 2>/tmp/eshkol_v12_compile_err; then
        # Run.  Some tests use (check-equal? ...) which prints "FAIL: ..."
        # and exits non-zero on mismatch; we treat both compile failure
        # and runtime non-zero as a failure.
        if /tmp/eshkol_v12_test >/tmp/eshkol_v12_run_out 2>&1; then
            echo "PASS"
            PASS=$((PASS + 1))
        else
            echo "FAIL (runtime)"
            head -10 /tmp/eshkol_v12_run_out | sed 's/^/    /'
            FAIL=$((FAIL + 1))
        fi
    else
        echo "FAIL (compile)"
        head -10 /tmp/eshkol_v12_compile_err | sed 's/^/    /'
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "Results: $PASS passed, $FAIL failed"
exit $FAIL
