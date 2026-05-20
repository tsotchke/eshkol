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

# Honour $BUILD_DIR (CI passes it via the matrix) so the asan-ubsan
# / xla / cuda lanes that build into build-asan / build-xla /
# build-cuda find their eshkol-run binary instead of falling through
# to a missing ./build/eshkol-run.
BUILD_DIR="${BUILD_DIR:-build}"
ESHKOL="./${BUILD_DIR}/eshkol-run"
PASS=0
FAIL=0
LOCK_DIR="${TMPDIR:-/tmp}/eshkol_v12_edge.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "Another v1.2 edge-case suite is already running; refusing to share temp/runtime resources." >&2
    exit 2
fi
TMP_WORK=$(mktemp -d "${TMPDIR:-/tmp}/eshkol_v12.XXXXXX")
RUN_OUT="$TMP_WORK/run.out"
COMPILE_ERR="$TMP_WORK/compile.err"
AOT_BIN="$TMP_WORK/aot-test"
cleanup() {
    rm -rf "$TMP_WORK" "$LOCK_DIR"
}
trap cleanup EXIT

has_failure_output() {
    grep -Eq '^[[:space:]]*FAIL:|RESULT: FAIL|RESULT: FAILURES|FAILURES DETECTED|SOME FAILED|SOME TESTS FAILED|TESTS FAILED' "$1"
}

show_failure_output() {
    grep -En '^[[:space:]]*FAIL:|RESULT: FAIL|RESULT: FAILURES|FAILURES DETECTED|SOME FAILED|SOME TESTS FAILED|TESTS FAILED' "$1" | head -10 | sed 's/^/    /'
}

# Some edge-case tests use `(load "tests/v1_2_edge_cases/foo.esk")` —
# project-root-relative paths.  resolve_module_path's first try is
# `<base_dir>/<path>`, where base_dir is the directory of the file
# being compiled (not cwd), so a relative path from project root
# only resolves if . is on $ESHKOL_PATH.  Add the project root by
# default so these tests run cleanly when invoked from the repo root.
export ESHKOL_PATH="${ESHKOL_PATH:-.}"

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
        if $ESHKOL -r "$test" >"$RUN_OUT" 2>&1; then
            if has_failure_output "$RUN_OUT"; then
                echo "FAIL (JIT assertion)"
                show_failure_output "$RUN_OUT"
                FAIL=$((FAIL + 1))
            else
                echo "PASS (JIT)"
                PASS=$((PASS + 1))
            fi
        else
            echo "FAIL (JIT)"
            head -10 "$RUN_OUT" | sed 's/^/    /'
            FAIL=$((FAIL + 1))
        fi
        continue
    fi

    # Compile (AOT)
    if $ESHKOL "$test" -o "$AOT_BIN" 2>"$COMPILE_ERR"; then
        # Run.  Some tests use (check-equal? ...) which prints "FAIL: ..."
        # and exits non-zero on mismatch; we treat both compile failure
        # and runtime non-zero as a failure.
        if "$AOT_BIN" >"$RUN_OUT" 2>&1; then
            if has_failure_output "$RUN_OUT"; then
                echo "FAIL (assertion)"
                show_failure_output "$RUN_OUT"
                FAIL=$((FAIL + 1))
            else
                echo "PASS"
                PASS=$((PASS + 1))
            fi
        else
            echo "FAIL (runtime)"
            head -10 "$RUN_OUT" | sed 's/^/    /'
            FAIL=$((FAIL + 1))
        fi
    else
        echo "FAIL (compile)"
        head -10 "$COMPILE_ERR" | sed 's/^/    /'
        FAIL=$((FAIL + 1))
    fi
done

# Shell-based regression tests in the same directory (compile-time
# diagnostics, etc. — anything that can't be expressed as an .esk
# program because the test is *about* a compile failure).
for test in tests/v1_2_edge_cases/*.sh; do
    [ -f "$test" ] || continue
    name=$(basename "$test" .sh)
    printf "  %-50s " "$name"
    if bash "$test" >"$RUN_OUT" 2>&1; then
        if has_failure_output "$RUN_OUT"; then
            echo "FAIL (shell assertion)"
            show_failure_output "$RUN_OUT"
            FAIL=$((FAIL + 1))
        else
            echo "PASS (shell)"
            PASS=$((PASS + 1))
        fi
    else
        echo "FAIL (shell)"
        head -10 "$RUN_OUT" | sed 's/^/    /'
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "Results: $PASS passed, $FAIL failed"
exit $FAIL
