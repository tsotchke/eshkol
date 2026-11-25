#!/usr/bin/env bash
# JIT and AOT load a sibling source through a directory containing dots.

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN="$ROOT/${BUILD_DIR:-build}/eshkol-run"

if [ ! -x "$RUN" ]; then
    echo "SKIP: $RUN not built"
    exit 0
fi

# Use a directory NAME with a literal dot in it.  mktemp -t puts
# things under $TMPDIR which on macOS already has dotted components,
# but be explicit so the test reproduces on Linux too where $TMPDIR
# is just /tmp.
WORK=$(mktemp -d -t "eshkol_load_dots.XXXXXX")
DOTTED="$WORK/dir.with.dots"
mkdir -p "$DOTTED"

trap 'rm -rf "$WORK"' EXIT

cat > "$DOTTED/lib.esk" <<EOF
(provide answer)
(define (answer) 42)
EOF

cat > "$WORK/main.esk" <<EOF
(load "dir.with.dots/lib.esk")
(display "answer: ") (display (answer)) (newline)
EOF

EXPECTED='answer: 42'

# JIT
JIT_OUT=$("$RUN" -r "$WORK/main.esk" 2>&1 | grep -E '^answer:' || true)
if [ "$JIT_OUT" != "$EXPECTED" ]; then
    echo "FAIL: JIT output mismatch"
    echo "  expected: $EXPECTED"
    echo "  actual:   $JIT_OUT"
    exit 1
fi

# AOT
AOT_BIN="$WORK/load_dots_aot"
"$RUN" "$WORK/main.esk" -o "$AOT_BIN" >/dev/null 2>&1
if [ ! -x "$AOT_BIN" ]; then
    echo "FAIL: AOT compile produced no binary"
    exit 1
fi
AOT_OUT=$("$AOT_BIN" 2>&1 | grep -E '^answer:' || true)
if [ "$AOT_OUT" != "$EXPECTED" ]; then
    echo "FAIL: AOT output mismatch"
    echo "  expected: $EXPECTED"
    echo "  actual:   $AOT_OUT"
    exit 1
fi

echo "PASS: (load \"...\") survives directory names containing dots"
exit 0
