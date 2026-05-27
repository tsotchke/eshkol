#!/usr/bin/env bash
set -euo pipefail

ESHKOL_RUN="${1:-${ESHKOL_RUN:-}}"
if [ -z "$ESHKOL_RUN" ]; then
    if [ -x "./build/eshkol-run" ]; then
        ESHKOL_RUN="./build/eshkol-run"
    elif [ -x "./build-verify/eshkol-run" ]; then
        ESHKOL_RUN="./build-verify/eshkol-run"
    else
        echo "FAIL: jit_cache_test could not locate eshkol-run" >&2
        exit 1
    fi
fi

if [ ! -x "$ESHKOL_RUN" ]; then
    echo "FAIL: jit_cache_test eshkol-run is not executable: $ESHKOL_RUN" >&2
    exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

src="$tmpdir/cache.esk"
cache="$tmpdir/cache"

printf '(display "cache-one") (newline)\n' > "$src"

ESHKOL_JIT_CACHE_DIR="$cache" ESHKOL_JIT_CACHE_TRACE=1 \
    "$ESHKOL_RUN" -r "$src" > "$tmpdir/out1" 2> "$tmpdir/err1"
grep -q '^cache-one$' "$tmpdir/out1"
grep -q '\[jit-cache\] miss ' "$tmpdir/err1"
grep -q '\[jit-cache\] store ' "$tmpdir/err1"

ESHKOL_JIT_CACHE_DIR="$cache" ESHKOL_JIT_CACHE_TRACE=1 \
    "$ESHKOL_RUN" -r "$src" > "$tmpdir/out2" 2> "$tmpdir/err2"
grep -q '^cache-one$' "$tmpdir/out2"
grep -q '\[jit-cache\] hit ' "$tmpdir/err2"

printf '(display "cache-two") (newline)\n' > "$src"
ESHKOL_JIT_CACHE_DIR="$cache" ESHKOL_JIT_CACHE_TRACE=1 \
    "$ESHKOL_RUN" -r "$src" > "$tmpdir/out3" 2> "$tmpdir/err3"
grep -q '^cache-two$' "$tmpdir/out3"
grep -q '\[jit-cache\] miss ' "$tmpdir/err3"
grep -q '\[jit-cache\] store ' "$tmpdir/err3"

entry_count="$(find "$cache" -type f | wc -l | tr -d ' ')"
if [ "$entry_count" -lt 2 ]; then
    echo "FAIL: expected at least two cache entries after source invalidation, found $entry_count" >&2
    exit 1
fi

echo "PASS: jit_cache_test"
