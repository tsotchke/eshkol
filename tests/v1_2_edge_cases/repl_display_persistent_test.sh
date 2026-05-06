#!/bin/bash
# repl_display_persistent_test.sh — displayed REPL expressions must not
# abort the process before the next read.

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REPL="$ROOT/${BUILD_DIR:-build}/eshkol-repl"

if [ ! -x "$REPL" ]; then
    echo "SKIP: $REPL not built"
    exit 0
fi

OUT=$(printf '(+ 1 2)\n(+ 3 4)\n' | "$REPL" --stdlib 2>&1)
RC=$?

if [ "$RC" -ne 0 ]; then
    echo "FAIL: eshkol-repl exited $RC after displayed expression"
    echo "$OUT" | sed 's/^/  /'
    exit 1
fi

if ! echo "$OUT" | grep -qx '3'; then
    echo "FAIL: first expression result missing"
    echo "$OUT" | sed 's/^/  /'
    exit 1
fi

if ! echo "$OUT" | grep -qx '7'; then
    echo "FAIL: second expression was not evaluated"
    echo "$OUT" | sed 's/^/  /'
    exit 1
fi

echo "PASS: eshkol-repl stays alive after displayed expressions"
exit 0
