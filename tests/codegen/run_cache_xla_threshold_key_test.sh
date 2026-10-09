#!/usr/bin/env bash
# tests/codegen/run_cache_xla_threshold_key_test.sh — the `-r` run cache must
# not hand back a binary compiled under a different ESHKOL_XLA_THRESHOLD.
#
# The XLA dispatch threshold is read at compile time and emitted as a
# constant, so it is part of what a cached binary *is*. Before the fix the
# cache key ignored it: a program compiled with the GPU cutoff at 100000 was
# silently reused when rerun with the cutoff raised past every tensor, i.e.
# a run meant to measure the CPU path executed the GPU-dispatching binary.
#
# Three runs of one source against a private cache directory:
#   1. threshold 100000        -> miss (compiles)
#   2. threshold 10^12         -> must MISS (a hit is the bug)
#   3. threshold 100000 again  -> must HIT  (the cache still works)
#
# Usage: tests/codegen/run_cache_xla_threshold_key_test.sh   (BUILD_DIR, default: build)
set -u
export LC_ALL=C LC_CTYPE=C LANG=C
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"
export ESHKOL_TEST_TMP_ROOT="${ESHKOL_TEST_TMP_ROOT:-$REPO_ROOT/.scratch}"
source "$REPO_ROOT/scripts/lib/test_isolation.sh" || { echo "FAIL: cannot source test isolation helper" >&2; exit 1; }
eshkol_test_isolation_init "run-cache-xla-threshold" || exit 1
BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in /*) B="$BUILD_DIR" ;; *) B="$REPO_ROOT/$BUILD_DIR" ;; esac
ESHKOL_RUN="${ESHKOL_RUN:-$B/eshkol-run}"
[ -x "$ESHKOL_RUN" ] || { echo "run_cache_xla_threshold_key_test.sh: $ESHKOL_RUN not found" >&2; exit 2; }

WORK="$ESHKOL_TEST_TMPDIR"
eshkol_test_toolchain_snapshot "$B" || { echo "FAIL: toolchain snapshot failed" >&2; exit 1; }
cat > "$WORK/prog.esk" <<'ESK'
(display (tensor-sum (tensor-add (make-tensor (list 8) 1.0) (make-tensor (list 8) 2.0))))
(newline)
ESK

run() {  # label threshold -> require successful program and exactly one valid cache status
    local label=$1 threshold=$2
    ESHKOL_JIT_CACHE_DIR="$WORK/cache" ESHKOL_JIT_CACHE_TRACE=1 ESHKOL_XLA_THRESHOLD="$threshold" \
        eshkol_test_timeout 300 "$ESHKOL_RUN" -r "$WORK/prog.esk" -L"$B" > "$WORK/$label.out" 2>&1 || {
            echo "FAIL: $label execution failed" >&2
            sed 's/^/  /' "$WORK/$label.out" >&2
            return 1
        }
    grep -Eq '^24(\.0)?$' "$WORK/$label.out" || {
        echo "FAIL: $label did not produce the expected tensor result" >&2
        sed 's/^/  /' "$WORK/$label.out" >&2
        return 1
    }
    local cache_line_count lookup_count store_count status recognized_count
    cache_line_count=$(grep -c '^\[jit-cache\] ' "$WORK/$label.out" || true)
    lookup_count=$(grep -E '^\[jit-cache\] (miss|hit)([[:space:]].*)?$' "$WORK/$label.out" | wc -l | tr -d ' ')
    store_count=$(grep -E '^\[jit-cache\] store([[:space:]].*)?$' "$WORK/$label.out" | wc -l | tr -d ' ')
    recognized_count=$((lookup_count + store_count))
    if [ "$lookup_count" -ne 1 ] || [ "$store_count" -gt 1 ] || [ "$cache_line_count" -ne "$recognized_count" ]; then
        echo "FAIL: $label expected one lookup plus at most one store (got lookups=$lookup_count stores=$store_count cache-lines=$cache_line_count)" >&2
        sed 's/^/  /' "$WORK/$label.out" >&2
        return 1
    fi
    status=$(grep -E '^\[jit-cache\] (miss|hit)([[:space:]].*)?$' "$WORK/$label.out" | sed -E 's/^\[jit-cache\] (miss|hit).*/\1/')
    printf '%s' "$status"
}

first=$(run first 100000) || exit 1
second=$(run second 1000000000000) || exit 1
third=$(run third 100000) || exit 1
if [ "$first" != "miss" ]; then
    echo "FAIL: run_cache_xla_threshold_key_test: first run reported '$first', expected a miss (is the run cache enabled?)"
    exit 1
fi
if [ "$second" = "hit" ]; then
    echo "FAIL: run_cache_xla_threshold_key_test: a binary compiled under ESHKOL_XLA_THRESHOLD=100000 was reused under 10^12"
    exit 1
fi
if [ "$third" != "hit" ]; then
    echo "FAIL: run_cache_xla_threshold_key_test: rerun under the original threshold reported '$third', expected a hit"
    exit 1
fi
if ! eshkol_test_toolchain_verify "$B"; then
    echo "FAIL: run_cache_xla_threshold_key_test: toolchain changed during run" >&2
    exit 1
fi
echo "PASS: run_cache_xla_threshold_key_test: miss / $second / hit"
