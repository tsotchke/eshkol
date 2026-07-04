#!/bin/sh
# Compile and run the Eshkol Taylor-tower Phase-0 POC.
# Self-contained: no repo dependencies. Exits 0 iff every check passes.
set -eu

here="$(cd "$(dirname "$0")" && pwd)"
CC="${CC:-cc}"
out="$here/taylor_poc"

"$CC" -O2 -std=c11 -Wall -Wextra "$here/taylor_poc.c" -lm -o "$out"
"$out"
status=$?
rm -f "$out"
exit "$status"
