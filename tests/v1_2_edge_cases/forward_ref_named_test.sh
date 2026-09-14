#!/usr/bin/env bash
# Bug W regression (2026-04-25).
# Assert that unresolved forward calls exit nonzero and name the function.
# Usage: forward_ref_named_test.sh [--self-test] [path/to/eshkol-run]

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$ROOT/tests/v1_2_edge_cases/forward_ref_named_test.sh"
SELF_TEST=0
if [ "${1:-}" = "--self-test" ]; then SELF_TEST=1; shift; fi

if [ -n "${1:-}" ]; then
    RUN="$1"
else
    BUILD_PATH="${BUILD_DIR:-build}"
    case "$BUILD_PATH" in
        /*|[A-Za-z]:/*) RUN="$BUILD_PATH/eshkol-run" ;;
        *) RUN="$ROOT/$BUILD_PATH/eshkol-run" ;;
    esac
fi

TMP_BASE="${ESHKOL_TEST_TMPDIR:-${ESHKOL_DURABLE_WORK_ROOT:-${ESHKOL_TEST_TMP_ROOT:-${TMPDIR:-/tmp}}}}"
WORK="$(mktemp -d "$TMP_BASE/eshkol-forward-ref.XXXXXX")" || {
    echo "INFRA: forward_ref_named_test could not create a temporary directory" >&2
    exit 125
}
cleanup() {
    if [ -n "${ESHKOL_DURABLE_WORK_ROOT:-}" ] ||
       [ -n "${ESHKOL_TEST_KEEP_TMPDIR:-}" ]; then
        echo "forward_ref_named_test: retaining temporary directory: $WORK" >&2
        return
    fi
    rm -rf -- "$WORK"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if [ "$SELF_TEST" -eq 1 ]; then
    FAKE_RUN="$WORK/fake-eshkol-run"
    cat > "$FAKE_RUN" <<'EOF'
#!/usr/bin/env bash
if [ "${FORWARD_REF_FAKE_MODE:-missing-name}" != "missing-name" ]; then
    case "$2" in
        */direct.esk) name=some-totally-undefined-fn ;;
        */nested.esk) name=deep-undefined-fn ;;
        *) name=meta-meta-cycle-style-name-with-dashes ;;
    esac
    echo "called undefined function '$name'" >&2
    if [ "$FORWARD_REF_FAKE_MODE" = "signal-exit" ]; then exit 139; fi
    exit 0
fi
echo "synthetic diagnostic without the unresolved function name" >&2
exit 1
EOF
    chmod +x "$FAKE_RUN"

    run_negative_control() {
        local mode="$1" required="$2" output rc
        if output=$(FORWARD_REF_FAKE_MODE="$mode" "$SCRIPT" "$FAKE_RUN" 2>&1); then
            rc=0
        else
            rc=$?
        fi
        [ "$rc" -eq 1 ] &&
            [ "$(printf '%s\n' "$output" | grep -c '^FAIL:')" -eq 3 ] &&
            printf '%s\n' "$output" | grep -Fq "$required"
    }

    if run_negative_control missing-name \
           "expected: called undefined function 'some-totally-undefined-fn'" &&
       run_negative_control zero-exit "exit: 0 (expected 1)" &&
       run_negative_control signal-exit "exit: 139 (expected 1)"; then
        echo "PASS: forward_ref_named_test negative controls (diagnostic, success, and signal exit)"
        exit 0
    fi
    echo "FAIL: forward_ref_named_test negative controls were not detected" >&2
    exit 1
fi

if [ ! -x "$RUN" ]; then
    echo "INFRA: forward_ref_named_test requires executable eshkol-run: $RUN" >&2
    exit 127
fi

PASS=0
FAIL=0

expect_named() {
    local id="$1" label="$2" script="$3" expected_name="$4"
    local stdout_file="$WORK/$id.stdout" stderr_file="$WORK/$id.stderr" rc

    if "$RUN" -r "$script" >"$stdout_file" 2>"$stderr_file"; then
        rc=0
    else
        rc=$?
    fi

    # A diagnosed rejection exits 1; 128+ means a signal/crash, not a pass.
    if [ "$rc" -eq 1 ] &&
       grep -Fq "called undefined function '$expected_name'" "$stderr_file"; then
        echo "PASS: $label"
        PASS=$((PASS + 1))
        return
    fi

    echo "FAIL: $label"
    echo "  expected: called undefined function '$expected_name'"
    echo "  exit: $rc (expected 1)"
    echo "  stderr (first 12 lines):"
    sed -n '1,12p' "$stderr_file" | sed 's/^/    /'
    FAIL=$((FAIL + 1))
}

cat > "$WORK/direct.esk" <<'EOF'
(require stdlib)
(some-totally-undefined-fn 42)
(display "should not reach") (newline)
EOF

cat > "$WORK/nested.esk" <<'EOF'
(require stdlib)
(define (helper x) (deep-undefined-fn x))
(helper 99)
EOF

cat > "$WORK/long-name.esk" <<'EOF'
(require stdlib)
(meta-meta-cycle-style-name-with-dashes '(a b c))
EOF

expect_named direct "direct call to undefined name" \
    "$WORK/direct.esk" "some-totally-undefined-fn"
expect_named nested "undefined name from inside helper" \
    "$WORK/nested.esk" "deep-undefined-fn"
expect_named long-name "long hyphenated name" \
    "$WORK/long-name.esk" "meta-meta-cycle-style-name-with-dashes"

echo
echo "Passed: $PASS  Failed: $FAIL"
if [ "$FAIL" -ne 0 ]; then exit 1; fi
exit 0
