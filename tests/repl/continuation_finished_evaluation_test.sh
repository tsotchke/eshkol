#!/bin/sh
# The interactive REPL runs each top-level form as its own evaluation. A
# continuation captured during one evaluation is resumed only while that
# evaluation is live: invoking it from a later form raises a catchable
# condition and the session carries on. Re-entry within a single evaluation
# stays multi-shot.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
REPL="$BUILD_DIR/eshkol-repl"
if [ ! -x "$REPL" ]; then echo "SKIP: $REPL not built"; exit 0; fi
out=$(printf '%s\n' \
    '(define k #f)' \
    '(define (f) (+ 1 (call/cc (lambda (c) (set! k c) 1))))' \
    '(display (f))' \
    '(k 10)' \
    '(display "|after")' \
    '(guard (e ((error-object? e) (display "|caught"))) (k 5))' \
    '(define (g) (let ((v (call/cc (lambda (c) (set! k c) 0)))) (if (< v 3) (k (+ v 1)) v)))' \
    '(display "|")' \
    '(display (g))' \
    | "$REPL" 2>&1)
status=$?
case "$out" in
    *"2"*"continuation cannot be resumed"*"|after|caught|3"*)
        if [ "$status" -eq 0 ]; then
            echo "PASS: REPL continuation from a finished evaluation raises"
            exit 0
        fi ;;
esac
echo "FAIL: REPL continuation from a finished evaluation (exit $status)"
printf '%s\n' "$out" | tail -20
exit 1
