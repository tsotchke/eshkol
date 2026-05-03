#!/bin/bash
# provide_aot_jit_parity_test.sh — Bug Z regression (filed 2026-04-30,
# closed by Eshkol 1235e0a).
#
# Pre-fix: `(provide pub)` was an informational declaration in JIT
# mode but a hard export boundary in AOT mode.  Code that worked
# under `eshkol-run -r` would `Unknown function: priv` under
# `eshkol-run` (AOT).  Noesis had 65 files using `(load ...)` to
# expose the full module, all of which silently broke under AOT.
#
# This test asserts that `provide` is informational in BOTH modes:
# `(load ...)` exposes everything in the file.  If a strict-export
# mode is desired in the future, it should be opted into via a
# separate keyword (e.g. `(export ...)`).

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN="$ROOT/${BUILD_DIR:-build}/eshkol-run"

if [ ! -x "$RUN" ]; then
    echo "SKIP: $RUN not built"
    exit 0
fi

WORK=$(mktemp -d -t eshkol_provide_parity_test.XXXXXX)
trap 'rm -rf "$WORK" /tmp/provide_parity_aot.$$' EXIT

cat > "$WORK/lib.esk" <<EOF
(provide pub)
(define (pub) 84)
(define (priv) 42)
EOF

cat > "$WORK/main.esk" <<EOF
(load "$WORK/lib.esk")
(display "priv-result: ") (display (priv)) (newline)
(display "pub-result: ") (display (pub)) (newline)
EOF

# Both modes must print the same two lines.  Under AOT a non-zero
# exit or "Unknown function: priv" indicates the regression has
# returned.
EXPECTED=$'priv-result: 42\npub-result: 84'

# JIT
JIT_OUT=$("$RUN" -r "$WORK/main.esk" 2>&1 | grep -E '^(priv|pub)-result' || true)
if [ "$JIT_OUT" != "$EXPECTED" ]; then
    echo "FAIL: JIT output mismatch"
    echo "  expected:"; echo "$EXPECTED" | sed 's/^/    /'
    echo "  actual:"  ; echo "$JIT_OUT"  | sed 's/^/    /'
    exit 1
fi

# AOT
AOT_BIN="/tmp/provide_parity_aot.$$"
"$RUN" "$WORK/main.esk" -o "$AOT_BIN" >/dev/null 2>&1
if [ ! -x "$AOT_BIN" ]; then
    echo "FAIL: AOT compile produced no binary"
    exit 1
fi
AOT_OUT=$("$AOT_BIN" 2>&1 | grep -E '^(priv|pub)-result' || true)
if [ "$AOT_OUT" != "$EXPECTED" ]; then
    echo "FAIL: AOT output mismatch (Bug Z regression!)"
    echo "  expected:"; echo "$EXPECTED" | sed 's/^/    /'
    echo "  actual:"  ; echo "$AOT_OUT"  | sed 's/^/    /'
    exit 1
fi

echo "PASS: provide is informational in both JIT and AOT"
exit 0
