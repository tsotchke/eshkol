#!/usr/bin/env bash
# The `-r` run cache must not reuse tree B's binary for tree A: x.esk in tree A
# embeds "\0dep-path\0<y.esk>\0dep-bytes\0", which NUL-delimited key framing
# serialized like tree B's separate x.esk and y.esk.
# Usage: tests/codegen/run_cache_key_framing_test.sh [eshkol-run]   (or BUILD_DIR, default: build)
set -u
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
export ESHKOL_TEST_TMP_ROOT="${ESHKOL_TEST_TMP_ROOT:-$REPO_ROOT/.scratch}"
source "$REPO_ROOT/scripts/lib/test_isolation.sh" || { echo "FAIL: cannot source test isolation helper" >&2; exit 1; }
eshkol_test_isolation_init "run-cache-key-framing" || exit 1
if [ -n "${1:-}" ]; then
    ESHKOL_RUN="$1"
else
    BUILD_DIR="${BUILD_DIR:-build}"
    case "$BUILD_DIR" in /*) ESHKOL_RUN="$BUILD_DIR/eshkol-run" ;; *) ESHKOL_RUN="$REPO_ROOT/$BUILD_DIR/eshkol-run" ;; esac
fi
[ -x "$ESHKOL_RUN" ] || { echo "FAIL: run_cache_key_framing_test: $ESHKOL_RUN not found" >&2; exit 2; }
B="$(cd "$(dirname "$ESHKOL_RUN")" && pwd)"

WORK="$ESHKOL_TEST_TMPDIR"
P="$WORK/p"
eshkol_test_toolchain_snapshot "$B" || { echo "FAIL: toolchain snapshot failed" >&2; exit 1; }

make_entry() {
    printf '(load "x.esk")\n(display "v=") (display v) (newline)\n' > "$P/e.esk"
}
make_tree_b() {
    rm -rf "$P" && mkdir -p "$P" && make_entry
    printf '(define v 1)\n; (load "y.esk") ;' > "$P/x.esk"
    printf '\n(define v 2)\n' > "$P/y.esk"
}
make_tree_a() {
    rm -rf "$P" && mkdir -p "$P" && make_entry
    local canonical_dir
    canonical_dir="$(cd "$P" && pwd -P)"
    {
        printf '(define v 1)\n; (load "y.esk") ;'
        printf '\0dep-path\0%s/y.esk\0dep-bytes\0' "$canonical_dir"
        printf '\n(define v 2)\n'
    } > "$P/x.esk"
}

run() {  # label expected-output -> prints the cache status (miss|hit)
    local label=$1 expected=$2
    (
        cd "$P" &&
        ESHKOL_LANGUAGE_COVERAGE_TRACE_DIR= \
        ESHKOL_JIT_CACHE_DIR="$WORK/cache" ESHKOL_JIT_CACHE_TRACE=1 \
            eshkol_test_timeout 300 "$ESHKOL_RUN" -r e.esk -L"$B"
    ) > "$WORK/$label.out" 2>&1 || {
        echo "FAIL: run_cache_key_framing_test: $label execution failed" >&2
        sed 's/^/  /' "$WORK/$label.out" >&2
        return 1
    }
    local lookups
    lookups=$(grep -E '^\[jit-cache\] (miss|hit)([[:space:]].*)?$' "$WORK/$label.out" | wc -l | tr -d ' ')
    if [ "$lookups" -ne 1 ]; then
        echo "FAIL: run_cache_key_framing_test: $label expected one cache lookup, got $lookups" >&2
        sed 's/^/  /' "$WORK/$label.out" >&2
        return 1
    fi
    if ! grep -qx "$expected" "$WORK/$label.out"; then
        echo "FAIL: run_cache_key_framing_test: $label did not print $expected" >&2
        sed 's/^/  /' "$WORK/$label.out" >&2
        return 1
    fi
    grep -E '^\[jit-cache\] (miss|hit)' "$WORK/$label.out" | sed -E 's/^\[jit-cache\] (miss|hit).*/\1/'
}

make_tree_b
first=$(run first v=1) || exit 1
make_tree_a
second=$(run second v=2)
second_status=$?
third=$(run third v=2) || exit 1
if [ "$first" != "miss" ]; then
    echo "FAIL: run_cache_key_framing_test: first run reported '$first', expected a miss (is the run cache enabled?)"
    exit 1
fi
if [ "$second_status" -ne 0 ] || [ "$second" = "hit" ]; then
    echo "FAIL: run_cache_key_framing_test: tree A reused the cached binary of tree B (status '$second')"
    sed 's/^/  /' "$WORK/second.out"
    exit 1
fi
if [ "$third" != "hit" ]; then
    echo "FAIL: run_cache_key_framing_test: rerun of tree A reported '$third', expected a hit"
    exit 1
fi
if ! eshkol_test_toolchain_verify "$B"; then
    echo "FAIL: run_cache_key_framing_test: toolchain changed during run" >&2
    exit 1
fi
echo "PASS: run_cache_key_framing_test: miss / $second / hit"
