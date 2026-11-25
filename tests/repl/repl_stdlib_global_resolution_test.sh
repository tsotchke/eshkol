#!/usr/bin/env bash
set -euo pipefail

ESHKOL_RUN="$1"
BUILD_DIR="$2"

export ESHKOL_LIB_DIR="$BUILD_DIR"

out1="$("$ESHKOL_RUN" -e '(display (null? stream-null)) (newline)' 2>&1)"
echo "$out1" | grep -qx '#t' || {
    echo "FAIL: direct stdlib data global stream-null did not evaluate to #t" >&2
    echo "$out1" >&2
    exit 1
}

out2="$("$ESHKOL_RUN" -e '(define s (stream-cons 42 (delay stream-null))) (display (stream-car s)) (display " ") (display (stream-null? (stream-cdr s))) (newline)' 2>&1)"
echo "$out2" | grep -qx '42 #t' || {
    echo "FAIL: stdlib data global inside delay closure failed" >&2
    echo "$out2" >&2
    exit 1
}

out3="$("$ESHKOL_RUN" -e '(define stream-null 99) (display stream-null) (newline)' 2>&1)"
echo "$out3" | grep -qx '99' || {
    echo "FAIL: user redefinition shadowing stdlib data global failed" >&2
    echo "$out3" >&2
    exit 1
}

echo "PASS: repl_stdlib_global_resolution_test"
