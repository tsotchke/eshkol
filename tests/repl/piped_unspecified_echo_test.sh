#!/bin/sh
# ADR-0023: the REPL shows nothing after a form that evaluates to the
# unspecified value, and still shows the empty list. Before the fix
# `(when (> x 1) (display "hi"))` echoed `hi()`.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
REPL="$BUILD_DIR/eshkol-repl"
if [ ! -x "$REPL" ]; then echo "SKIP: $REPL not built"; exit 0; fi
got=$(printf '(define x 5)\n(when (> x 1) (display "hi"))\n(display "Hello")\n(newline)\n(+ 1 2)\n(set! x 6)\n(quote ())\n(if #f 1)\n(vector-set! (vector 1) 0 2)\n' | "$REPL" 2>/dev/null | od -c | tr -s ' ' | tr '\n' ' ')
want=$(printf 'hiHello\n3\n()\n' | od -c | tr -s ' ' | tr '\n' ' ')
if [ "$got" = "$want" ]; then
    echo "PASS: piped REPL prints nothing for the unspecified value"
    exit 0
fi
echo "FAIL: piped REPL echo"
echo "  want: $want"
echo "  got:  $got"
exit 1
