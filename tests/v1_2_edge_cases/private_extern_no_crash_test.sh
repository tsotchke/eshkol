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

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/eshkol-private-extern.XXXXXX") || exit 1
case "$tmp_dir" in
    "${TMPDIR:-/tmp}"/eshkol-private-extern.*) ;;
    *) echo "FAIL: unexpected temp dir: $tmp_dir"; exit 1 ;;
esac
cleanup() {
    [ -n "${tmp_dir:-}" ] && [ -d "$tmp_dir" ] && rm -rf -- "$tmp_dir"
}
trap cleanup EXIT

# Make a throwaway stdlib module that hits the old crash pattern,
# then have a user file require it.
mkdir -p "$tmp_dir/core"
mod_file="$tmp_dir/core/_extern_rename_test_mod.esk"
cat > "$mod_file" <<'ESK'
(require core.list.transform)
(provide pub-fn)
(define pub-fn (lambda () 'public))
;; Private extern + private function that uses it. Pre-fix: crash here.
(extern void some-c-helper :real strlen)
(define (priv-helper) (some-c-helper))
ESK
use_file="$tmp_dir/extern_rename_test_use.esk"
cat > "$use_file" <<'ESK'
(require core._extern_rename_test_mod)
(display "ok")
(newline)
ESK
bin_file="$tmp_dir/extern_rename_test_bin"
"$RUN" "$use_file" -o "$bin_file" >/dev/null 2>&1
RC=$?
if [ "$RC" -eq 0 ]; then
    echo "PASS: precompiled module with private extern+ref compiles (rc=$RC)"
    PASS=$((PASS+1))
else
    echo "FAIL: precompiled module with private extern+ref crashed or failed (rc=$RC)"
    FAIL=$((FAIL+1))
fi

# The real-world trigger: collections_test.esk and cache_test.esk both
# require core.testing (which has a private extern). Pre-fix: 139.
for esk in tests/v1_2_edge_cases/collections_test.esk tests/v1_2_edge_cases/cache_test.esk; do
    name=$(basename "$esk" .esk)
    bin_file="$tmp_dir/${name}_xext_bin"
    "$RUN" "$esk" -o "$bin_file" >/dev/null 2>&1
    RC=$?
    if [ "$RC" -eq 0 ]; then
        echo "PASS: $name compiles cleanly (rc=$RC)"
        PASS=$((PASS+1))
    else
        echo "FAIL: $name crashed or failed (rc=$RC)"
        FAIL=$((FAIL+1))
    fi
done

echo
echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" -gt 0 ] && exit 1
exit 0
