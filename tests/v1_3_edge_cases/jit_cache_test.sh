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

# ── ESH-0183: editing a (load …)ed DEPENDENCY must invalidate the run-cache ──
# The key previously hashed only the entry file, so editing a transitively
# loaded file left the key unchanged and `-r` re-ran a STALE cached binary.
# entry.esk loads mid.esk which loads deep.esk; we edit the *deepest* file.
depdir="$(mktemp -d)"
mkdir -p "$depdir/cache"
printf '(define (deep-msg) "DEP_LOADED_ONE")\n' > "$depdir/deep.esk"
printf '(load "deep.esk")\n(define (mid-msg) (deep-msg))\n' > "$depdir/mid.esk"
printf '(load "mid.esk")\n(display (mid-msg)) (newline)\n' > "$depdir/entry.esk"

ESHKOL_JIT_CACHE_DIR="$depdir/cache" ESHKOL_JIT_CACHE_TRACE=1 \
    "$ESHKOL_RUN" -r "$depdir/entry.esk" > "$depdir/dout1" 2> "$depdir/derr1"
grep -q '^DEP_LOADED_ONE$' "$depdir/dout1" || { echo "FAIL: dep run1 wrong output" >&2; exit 1; }
grep -q '\[jit-cache\] store ' "$depdir/derr1" || { echo "FAIL: dep run1 did not store" >&2; exit 1; }

# Unchanged re-run: the cache must still HIT (the fix must not disable caching).
ESHKOL_JIT_CACHE_DIR="$depdir/cache" ESHKOL_JIT_CACHE_TRACE=1 \
    "$ESHKOL_RUN" -r "$depdir/entry.esk" > "$depdir/dout2" 2> "$depdir/derr2"
grep -q '\[jit-cache\] hit ' "$depdir/derr2" || { echo "FAIL: dep unchanged re-run should hit cache" >&2; exit 1; }
grep -q '^DEP_LOADED_ONE$' "$depdir/dout2" || { echo "FAIL: dep run2 wrong output" >&2; exit 1; }

# Edit the DEEPEST dependency only — entry.esk and mid.esk are untouched.
printf '(define (deep-msg) "DEP_LOADED_TWO")\n' > "$depdir/deep.esk"
ESHKOL_JIT_CACHE_DIR="$depdir/cache" ESHKOL_JIT_CACHE_TRACE=1 \
    "$ESHKOL_RUN" -r "$depdir/entry.esk" > "$depdir/dout3" 2> "$depdir/derr3"
grep -q '\[jit-cache\] miss ' "$depdir/derr3" || { echo "FAIL: editing a loaded dependency did not invalidate the cache (ESH-0183 regression)" >&2; exit 1; }
grep -q '^DEP_LOADED_TWO$' "$depdir/dout3" || { echo "FAIL: STALE output after dependency edit — cache shipped old code (ESH-0183 regression)" >&2; exit 1; }
rm -rf "$depdir"

# ── ESH-0183: the AOT (`-o`/--emit-object) path must reflect a fresh edit ─────
# A single fresh compile is uncached, but guard against any future object cache
# that could resurrect the "stale object on source edit" report.
aotdir="$(mktemp -d)"
printf '(define (aot-fn) (display "AOT_MARK_ORIG") (newline))\n(aot-fn)\n' > "$aotdir/a.esk"
"$ESHKOL_RUN" --emit-object -o "$aotdir/a.o" "$aotdir/a.esk" > "$aotdir/aot1.log" 2>&1 || { echo "FAIL: AOT compile 1 failed" >&2; cat "$aotdir/aot1.log" >&2; exit 1; }
strings "$aotdir/a.o" | grep -q 'AOT_MARK_ORIG' || { echo "FAIL: AOT object 1 missing its own string" >&2; exit 1; }
# Add a new top-level-reachable string near the end of the function, recompile.
printf '(define (aot-fn) (display "AOT_MARK_ORIG") (newline) (display "AOT_MARK_NEW_XYZ") (newline))\n(aot-fn)\n' > "$aotdir/a.esk"
"$ESHKOL_RUN" --emit-object -o "$aotdir/a.o" "$aotdir/a.esk" > "$aotdir/aot2.log" 2>&1 || { echo "FAIL: AOT compile 2 failed" >&2; cat "$aotdir/aot2.log" >&2; exit 1; }
strings "$aotdir/a.o" | grep -q 'AOT_MARK_NEW_XYZ' || { echo "FAIL: AOT object did not reflect the fresh edit (stale object regression)" >&2; exit 1; }
rm -rf "$aotdir"

echo "PASS: jit_cache_test"
