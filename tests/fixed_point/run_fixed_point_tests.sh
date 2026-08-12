#!/usr/bin/env bash
# run_fixed_point_tests.sh — build + run the fixed-point / i128 / exact-accumulation
# suite standalone (no CMake, no dependency on the Eshkol build tree). Additive:
# touches nothing outside this directory + the library it tests.
#
# Usage:  tests/fixed_point/run_fixed_point_tests.sh [--bench]
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
LIB="$ROOT/lib/math/fixed_point"
CC="${CC:-clang}"
CFLAGS="-std=c11 -Wall -Wextra -O2 -I$LIB -I$HERE"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; NC=$'\033[0m'
pass=0; fail=0; failed=()

build_run() {
    local name="$1"
    if ! "$CC" $CFLAGS "$HERE/$name.c" "$LIB/eshkol_fixed_point.c" -lm -o "$OUT/$name" 2>"$OUT/$name.err"; then
        echo "${RED}BUILD FAIL${NC}: $name"; cat "$OUT/$name.err"; fail=$((fail+1)); failed+=("$name(build)"); return
    fi
    if "$OUT/$name"; then pass=$((pass+1)); else fail=$((fail+1)); failed+=("$name"); fi
    echo
}

echo "========================================="
echo "  Eshkol fixed-point / i128 / dot_exact"
echo "  compiler: $($CC --version | head -1)"
echo "========================================="
echo

build_run test_i128
build_run test_fixed
build_run test_dot_exact

if [ "${1:-}" = "--bench" ]; then
    echo "----- benchmark -----"
    if ! "$CC" -std=c11 -Wall -Wextra -O3 -I"$LIB" -I"$HERE" \
        "$HERE/bench_dot_exact.c" "$LIB/eshkol_fixed_point.c" -lm \
        -o "$OUT/bench" 2>"$OUT/bench_dot_exact.err"; then
        echo "${RED}BUILD FAIL${NC}: bench_dot_exact"
        cat "$OUT/bench_dot_exact.err"
        fail=$((fail+1))
        failed+=("bench_dot_exact(build)")
    elif "$OUT/bench"; then
        pass=$((pass+1))
    else
        fail=$((fail+1))
        failed+=("bench_dot_exact")
    fi
    echo
fi

echo "========================================="
if [ "$fail" -eq 0 ]; then
    echo "${GREEN}ALL SUITES PASSED${NC} ($pass suites)"
    exit 0
else
    echo "${RED}FAILURES${NC}: ${failed[*]}"
    exit 1
fi
