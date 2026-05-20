#!/bin/bash
# forward_ref_named_test.sh — Bug W regression (2026-04-25)
#
# Asserts that calling a forward-referenced function that was never
# defined produces an error message that NAMES the function and
# exits non-zero.
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN="$ROOT/${BUILD_DIR:-build}/eshkol-run"

if [ ! -x "$RUN" ]; then
    echo "SKIP: $RUN not built"
    exit 0
fi

mk() { cat > "$1"; }

PASS=0
FAIL=0

# Helper: run a script in REPL/JIT mode, capture stderr+exit, check
# expected substrings appear and exit is non-zero.
expect_named() {
    local label="$1"
    local script="$2"
    local expected_name="$3"

    local err rc
    err=$("$RUN" -r "$script" 2>&1 1>/dev/null)
    rc=$?

    local ok_msg="no"
    if echo "$err" | grep -q "called undefined function '$expected_name'"; then
        ok_msg="yes"
    fi

    if [ "$ok_msg" = "yes" ] && [ "$rc" -ne 0 ]; then
        echo "PASS: $label"
        PASS=$((PASS+1))
    else
        echo "FAIL: $label (msg-named=$ok_msg, exit=$rc)"
        echo "  stderr: $err"
        FAIL=$((FAIL+1))
    fi
}

# ── Test 1: undefined function called directly ──────────────────
mk /tmp/fwd_t1.esk <<'EOF'
(require stdlib)
(some-totally-undefined-fn 42)
(display "should not reach") (newline)
EOF
expect_named "direct call to undefined name"  /tmp/fwd_t1.esk  "some-totally-undefined-fn"

# ── Test 2: undefined function called inside a helper ───────────
mk /tmp/fwd_t2.esk <<'EOF'
(require stdlib)
(define (helper x) (deep-undefined-fn x))
(helper 99)
EOF
expect_named "undefined name from inside helper"  /tmp/fwd_t2.esk  "deep-undefined-fn"

# ── Test 3: undefined function with a hyphenated/long name ──────
mk /tmp/fwd_t3.esk <<'EOF'
(require stdlib)
(meta-meta-cycle-style-name-with-dashes '(a b c))
EOF
expect_named "long hyphenated name"  /tmp/fwd_t3.esk  "meta-meta-cycle-style-name-with-dashes"

echo
echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" -gt 0 ] && exit 1
exit 0
