#!/bin/bash
# load_path_with_dots_test.sh — regression for the resolveModulePath
# bug where directory components containing dots were silently mangled.
#
# Pre-fix: parser stripped `.esk` and rewrote `/` → `.`, then
# resolveModulePath rewrote `.` → `/` — a roundtrip that worked for
# `lib/foo.esk` but corrupted any path whose directory components
# contained dots.  In particular every macOS $TMPDIR follows the
# pattern /var/folders/<2-letter>/<22-char>.<random>/T, so loading
# from $TMPDIR appeared to fail with a confusing
#   "Module not found: .var.folders.7b...eshkol_test.UlE2WwxyYH.lib"
# even though the file existed.
#
# Post-fix: the parser keeps string-literal load arguments verbatim
# and resolveModulePath detects path-like inputs (start with /, ./,
# ../, contain /, or end with .esk) and skips the dot-rewrite.
#
# This test creates a directory whose name contains a dot, drops a
# library file in it, and asserts that `(load "$WORK/lib.esk")` works
# under both JIT and AOT.

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

trap 'rm -rf "$WORK" /tmp/load_dots_aot.$$' EXIT

cat > "$DOTTED/lib.esk" <<EOF
(provide answer)
(define (answer) 42)
EOF

cat > "$WORK/main.esk" <<EOF
(load "$DOTTED/lib.esk")
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
AOT_BIN="/tmp/load_dots_aot.$$"
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
