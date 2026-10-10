#!/bin/sh
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
RUN="$BUILD_DIR/eshkol-run"
[ -x "$RUN" ] || { echo "SKIP: $RUN not built"; exit 0; }
TMP_BASE="${TMPDIR:-/tmp}"
tmp_dir=$(mktemp -d "$TMP_BASE/eshkol-import-scope-aot.XXXXXX") || exit 1
case "$tmp_dir" in
    "$TMP_BASE"/eshkol-import-scope-aot.*) ;;
    *) echo "FAIL: unexpected temp dir: $tmp_dir"; exit 1 ;;
esac
trap 'rm -rf -- "$tmp_dir"' EXIT HUP INT TERM
export ESHKOL_PATH="$ROOT/lib"
export ESHKOL_LIB_DIR="$BUILD_DIR"
"$RUN" -o "$tmp_dir/import_scope_alias_aot" \
    "$ROOT/tests/repl/import_scope_alias_test.esk"
"$tmp_dir/import_scope_alias_aot"
