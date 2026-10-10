#!/bin/sh
# `:type <expr>` prints the HoTT type checker's inferred type for the form,
# with the session's own definitions in scope.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
REPL="$BUILD_DIR/eshkol-repl"
if [ ! -x "$REPL" ]; then echo "SKIP: $REPL not built"; exit 0; fi
got=$(printf '(define (sq x) (* x x))\n:type 42\n:type 3.5\n:type "s"\n:type (+ 1 2.0)\n:type (list 1 2)\n:type sq\n:type (lambda ((x : integer)) x)\n' \
    | ESHKOL_PATH="$ROOT/lib" "$REPL" 2>/dev/null | sed -n 's/^.*Type: *//p' | sed 's/\x1b\[[0-9;]*m//g' | tr '\n' '|')
want='Int64|Float64|String|Float64|List|(-> Value Value)|(-> Int64 Int64)|'
if [ "$got" = "$want" ]; then
    echo "PASS: :type prints the inferred type"
    exit 0
fi
echo "FAIL: :type output"
echo "  want: $want"
echo "  got:  $got"
exit 1
