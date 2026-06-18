#!/usr/bin/env bash
# ESH-0039: runtime type errors must carry a "file:line:col:" source-location
# prefix pointing at the failing expression.
set -euo pipefail

ESHKOL_RUN="${1:-${ESHKOL_RUN:-}}"
if [ -z "$ESHKOL_RUN" ]; then
    if [ -x "./build/eshkol-run" ]; then
        ESHKOL_RUN="./build/eshkol-run"
    elif [ -x "./build-verify/eshkol-run" ]; then
        ESHKOL_RUN="./build-verify/eshkol-run"
    else
        echo "FAIL: source_span_type_error_test could not locate eshkol-run" >&2
        exit 1
    fi
fi

if [ ! -x "$ESHKOL_RUN" ]; then
    echo "FAIL: source_span_type_error_test eshkol-run is not executable: $ESHKOL_RUN" >&2
    exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

src="$tmpdir/span.esk"
bin="$tmpdir/span"

# The (+ x "bad") application is on line 1; the operator opens at column 14.
printf '(define (f x) (+ x "bad"))\n(f 1)\n' > "$src"

# --- AOT path -------------------------------------------------------------
if ! "$ESHKOL_RUN" "$src" -o "$bin" -L./build > "$tmpdir/compile.log" 2>&1; then
    echo "FAIL: AOT compile failed" >&2
    cat "$tmpdir/compile.log" >&2
    exit 1
fi

# Binary must exit non-zero (unhandled type error) and print a prefixed message.
set +e
"$bin" > "$tmpdir/out" 2> "$tmpdir/err"
status=$?
set -e

if [ "$status" -eq 0 ]; then
    echo "FAIL: AOT binary exited 0 but a type error was expected" >&2
    exit 1
fi

# Expect: "<...>span.esk:1:<col>: ...Type error.../...operand is not a number..."
if ! grep -Eq 'span\.esk:1:[0-9]+: ' "$tmpdir/err"; then
    echo "FAIL: AOT type error message lacks the 'file:line:col:' prefix" >&2
    echo "  got: $(cat "$tmpdir/err")" >&2
    exit 1
fi

echo "PASS: source_span_type_error_test (AOT) -> $(cat "$tmpdir/err")"
exit 0
