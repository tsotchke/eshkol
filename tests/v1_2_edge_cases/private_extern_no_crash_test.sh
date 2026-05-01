#!/bin/bash
# private_extern_no_crash_test.sh — EXTERN_OP rename SIGSEGV (2026-04-26)
#
# Regression for the union-aliasing bug in update_ast_references:
# precompiled modules that have BOTH a (provide ...) list and a private
# (define ...) referencing an (extern ...) declaration crashed the
# compiler at SIGSEGV (139) during process_requires's symbol-rename pass.
# The fix routes ESHKOL_EXTERN_OP through extern_op (correct union slot)
# instead of call_op.
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN="$ROOT/${BUILD_DIR:-build}/eshkol-run"
[ ! -x "$RUN" ] && { echo "SKIP: $RUN not built"; exit 0; }

PASS=0; FAIL=0

# Make a throwaway stdlib module that hits the old crash pattern,
# then have a user file require it.
MOD="$ROOT/lib/core/_extern_rename_test_mod.esk"
cat > "$MOD" <<'ESK'
(require core.list.transform)
(provide pub-fn)
(define pub-fn (lambda () 'public))
;; Private extern + private function that uses it. Pre-fix: crash here.
(extern void some-c-helper :real strlen)
(define (priv-helper) (some-c-helper))
ESK
USE=/tmp/extern_rename_test_use.esk
cat > "$USE" <<'ESK'
(require core._extern_rename_test_mod)
(display "ok")
(newline)
ESK
BIN=/tmp/extern_rename_test_bin
rm -f "$BIN"
"$RUN" "$USE" -o "$BIN" >/dev/null 2>&1
RC=$?
rm -f "$MOD"
if [ "$RC" -ne 139 ]; then
    echo "PASS: precompiled module with private extern+ref doesn't SIGSEGV (rc=$RC)"
    PASS=$((PASS+1))
else
    echo "FAIL: still SIGSEGVs"
    FAIL=$((FAIL+1))
fi

# The real-world trigger: collections_test.esk and cache_test.esk both
# require core.testing (which has a private extern). Pre-fix: 139.
for esk in tests/v1_2_edge_cases/collections_test.esk tests/v1_2_edge_cases/cache_test.esk; do
    name=$(basename "$esk" .esk)
    BIN=/tmp/${name}_xext_bin
    rm -f "$BIN"
    "$RUN" "$esk" -o "$BIN" >/dev/null 2>&1
    RC=$?
    if [ "$RC" -ne 139 ]; then
        echo "PASS: $name compiles without SIGSEGV (rc=$RC)"
        PASS=$((PASS+1))
    else
        echo "FAIL: $name SIGSEGV"
        FAIL=$((FAIL+1))
    fi
done

echo
echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" -gt 0 ] && exit 1
exit 0
