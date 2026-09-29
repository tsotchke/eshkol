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
BUILD_DIR="${BUILD_DIR:-build}"
case "$BUILD_DIR" in /*) B="$BUILD_DIR" ;; *) B="$REPO_ROOT/$BUILD_DIR" ;; esac
ESHKOL_RUN="${ESHKOL_RUN:-$B/eshkol-run}"
[ -x "$ESHKOL_RUN" ] || { echo "run_cache_xla_threshold_key_test.sh: $ESHKOL_RUN not found" >&2; exit 2; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cat > "$WORK/prog.esk" <<'ESK'
(display (tensor-sum (tensor-add (make-tensor (list 8) 1.0) (make-tensor (list 8) 2.0))))
(newline)
ESK

run() {  # threshold -> prints the [jit-cache] status word (hit/miss/...)
    ESHKOL_JIT_CACHE_DIR="$WORK/cache" ESHKOL_JIT_CACHE_TRACE=1 ESHKOL_XLA_THRESHOLD="$1" \
        timeout 300 "$ESHKOL_RUN" -r "$WORK/prog.esk" -L"$B" 2>&1 \
        | sed -n 's/^\[jit-cache\] \([a-z-]*\).*/\1/p' | head -1
}

first=$(run 100000)
second=$(run 1000000000000)
third=$(run 100000)
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
echo "PASS: run_cache_xla_threshold_key_test: miss / $second / hit"
