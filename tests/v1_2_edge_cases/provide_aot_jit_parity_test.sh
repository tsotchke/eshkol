#!/usr/bin/env bash
# provide_aot_jit_parity_test.sh — `load` remains an inline inclusion form.
#
# `provide` establishes an export boundary for `(require ...)` and
# `(import ...)`, but `(load ...)` deliberately includes a file in the
# current top-level environment. This test protects that distinction
# in both native execution modes.
#
# Keep all run artifacts in the repository's isolated scratch root.

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN="$ROOT/${BUILD_DIR:-build}/eshkol-run"

if [ ! -x "$RUN" ]; then
    echo "SKIP: $RUN not built"
    exit 0
fi

export ESHKOL_TEST_TMP_ROOT="$ROOT/.scratch"
ISO="$ROOT/scripts/lib/test_isolation.sh"
[ -r "$ISO" ] || { echo "FAIL: cannot read $ISO" >&2; exit 1; }
source "$ISO"
eshkol_test_isolation_init "provide-aot-jit-parity"
trap eshkol_test_isolation_cleanup EXIT

WORK="$ESHKOL_TEST_TMPDIR/work"
mkdir -p "$WORK"

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

# Both modes must print the same two lines because load has no module
# boundary. The require/import boundary is covered by visibility tests.
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
AOT_BIN="$WORK/provide_parity_aot"
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

echo "PASS: load remains inline in both JIT and AOT"
exit 0
