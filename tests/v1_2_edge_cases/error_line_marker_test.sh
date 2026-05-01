#!/bin/bash
# error_line_marker_test.sh — verify that compile-time error markers
# point at the actual source line/column.
#
# Bug history: `eshkol_parse_next_ast_from_stream` used to strip
# comment lines INCLUDING their trailing newline (via std::getline),
# and started a fresh tokenizer at line=1 for every form. The result:
# all compile errors were reported as line 1 (or wherever the most
# recent stdlib AST happened to be), regardless of the actual error
# site. Fixed by preserving newlines from comments and by tracking
# cumulative file line/column across successive parse calls.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="$ROOT/${BUILD_DIR:-build}/eshkol-run"

if [ ! -x "$RUN" ]; then
    echo "ERROR: $RUN not built"
    exit 1
fi

PASS=0
FAIL=0

# expect_err <case-name> <esk-source> <expected-substring> [--with-stdlib]
expect_err() {
    local name="$1"
    local src="$2"
    local expected="$3"
    local stdlib_flag="--no-stdlib"
    if [ "${4:-}" = "--with-stdlib" ]; then
        stdlib_flag=""
    fi
    local f
    f=$(mktemp /tmp/eshkol_err_marker_XXXXXX.esk)
    printf '%s\n' "$src" > "$f"
    local out
    out=$("$RUN" $stdlib_flag "$f" -o /tmp/eshkol_err_marker_out 2>&1 || true)
    if grep -qF "$expected" <<<"$out"; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        echo "FAIL: $name"
        echo "  expected substring: $expected"
        echo "  got:"
        echo "$out" | head -5 | sed 's/^/    /'
    fi
    rm -f "$f"
}

# Case 1: top-of-file error
expect_err "top-of-file error" \
'(undefined-fn 1 2 3)' \
':1:2:'

# Case 2: error after a single comment line (regression for the
# std::getline-eats-newline bug).
expect_err "error after comment" \
';; comment line one
(undefined-fn 1 2 3)' \
':2:2:'

# Case 3: error after several blank/comment/code lines.
expect_err "error after multiple lines" \
';; comment
(define x 1)
(define y 2)

(undefined-fn 1 2 3)' \
':5:2:'

# Case 4: error inside a multi-line define body — column is whatever
# the call sits at (4 for two-space-indented `  (`).
expect_err "error in nested function body" \
';; comment
(define (broken)
  (undefined-fn 1 2 3))' \
':3:4:'

# Case 5: with stdlib loaded — verifies the cumulative line counter
# isn't perturbed by the parser walking through multiple stdlib
# modules before reaching the user's program.  Pre-fix, the user's
# error would be reported at "line 1 of the last stdlib AST" or
# similar.
expect_err "error after stdlib load + user defines" \
';; uses stdlib so we exercise the stdlib parse path
(define xs (list 1 2 3))
(display (length xs)) (newline)

(undefined-fn)' \
':5:2:' --with-stdlib

echo "error-line-marker: $PASS pass, $FAIL fail"
exit $FAIL
